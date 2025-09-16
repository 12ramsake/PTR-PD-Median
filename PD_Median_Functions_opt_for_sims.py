import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

## The different considered mu and sigma in the paper

# location m estimate
# x is a univariate numpy array of size n
# def med(x):
#     return cp.median(x)

def med(x):
    return cp.percentile(x, 50)


# scale  m estimate
# x is a univariate numpy array of size n
def mad(x):
    return med(cp.abs(x-med(x)))

# scale  m estimate
# x is a univariate numpy array of size n
def iqr(x):
    q1 = cp.percentile(x, 25)
    q3 = cp.percentile(x, 75)
    iqr = q3 - q1
    return iqr

### Huber estimate (newton raps)

# x: real value
# beta: threshold param
def psi_huber(x,beta):
    abss=cp.abs(x)
    blc=abss>beta
    abss[blc]=beta
    return abss

# x: real value
# beta: threshold param
def psi_prime_huber(x,beta):
    bl=cp.abs(x)<beta
    return bl.astype(float)

#equation 3.1 in M estimator section 
def f_huber(y,x,scale,beta):
    step1=y-x
    signs=cp.sign(step1)
    return cp.mean(signs*psi_huber(signs*step1/scale,beta))

#deriv of equation 3.1 in M estimator section 
def f_prime_huber(y,x,scale,beta):
    abss=cp.abs(y-x)
    return cp.mean(psi_prime_huber(abss/scale,beta))

def huber(x,scale=None,beta=2.0,num_iter=40,tol=0.0001):
    if scale is None:
        scale=mad(x)
    #initial value
    # est=med(x)
    est=0
    # est=cp.ones(len(x))
    #NR
    for i in range(0,num_iter):
        val=0.5*f_huber(est,x,scale,beta)
        est=est- val
        if cp.abs(val)<tol:
            break
    return est

#assumes x sorted
def huber_joint(x,n,beta=2.0,num_iter=40,tol=0.0001):
    a=int(cp.floor((n+1)/2))
    scale=order_stat(cp.abs(x-x[a]),a)
    # initial value
    # est=med(x)
    est=0
    # est=cp.ones(len(x))
    #NR
    for i in range(0,num_iter):
        val=0.5*f_huber(est,x,scale,beta)
        est=est- val
        if cp.abs(val)<tol:
            break
    return cp.array([est,scale])


# dat=np.random.standard_normal(100)
# dat=cp.asarray(dat)
# dat[22]=100
# print(cp.mean(dat))
# huber(dat)
## Functions for the exponential mechanism
# Generate nvec unit vectors in dimension d
# d: dimension
# nvec: num unit vectors N
# mrng: object from mtalg package for seed
def generate_UV(nvec,d,mrng):
    u=mrng.standard_normal(size=(nvec,d))
    u=cp.asarray(u)
    for i in range(0,nvec):
        u[i,:]=u[i,:]/cp.linalg.norm(u[i,:])
    return u




# Computes the projections of the data onto the unit vectors, and the location and scale estimates for all u
# u: nvec x d numpy array of uniformly sampled unit vectors
# data: n  x d numpy array of data
# mu,sigma are the location and scale functions
def get_initial_values(u,data,mu,sigma):
    pj_data=cp.matmul(data,u.T)
    n=data.shape[0]
    # print(pj_data.shape)
    # pj_data=cp.apply_along_axis(cp.sort,axis=0,arr=pj_data)
    pj_data=cp.sort(pj_data,axis=0)
    if mu==huber and sigma==mad:
        scale_huber=cp.apply_along_axis(huber_joint,axis=0,arr=pj_data,n=n)
        # print(scale_huber.shape)
        loc=scale_huber[0,:]
        scale=scale_huber[1,:]
    else:
        if mu==med:
            loc=pj_data[int(cp.floor((n+1)/2)),:]
        else:
            loc=cp.apply_along_axis(mu,axis=0,arr=pj_data)
        if sigma==mad:
            a=int(cp.floor((n+1)/2))
            scale=cp.partition(cp.abs(pj_data-pj_data[a,:]),a-1 ,axis=0)[a-1,:]
        elif sigma==iqr:
            scale=pj_data[int(cp.floor(3*n/4)),:]-pj_data[int(cp.floor(n/4)),:]
        else:
            scale=cp.apply_along_axis(sigma,axis=0,arr=pj_data)
    # print(pj_data.shape)
    return loc, scale, pj_data

    

# Computes the outlyingness 
# x: d x 1 numpy array for point at which depth is computed
# u: nvec x d numpy array of uniformly sampled unit vectors
# loc,scale: nvec x 1 numpy array of location and scale estimates for each u
def outlyingness(x,loc,scale,u,pj_x=None):
    if pj_x is None:
        pj_x=cp.matmul(u,cp.reshape(x,(d,1)))
    # Now for each u find univariate depth and average, u are the columns
    Ou=cp.abs(cp.squeeze(pj_x)-loc)/scale
    O=cp.max(Ou)
    ux=cp.argmax(Ou)
    return O,ux


# Computes the gradient of the outlyingness function
# x: d x 1 numpy array for point at which depth is computed
# u: nvec x d numpy array of uniformly sampled unit vectors
# loc,scale: nvec x 1 numpy array of location and scale estimates for each u
def grad_outlyingness(x,loc,scale,u,d,gd=True):
    # x=cupy.asarray(x)
    pj_x=cp.matmul(u,cp.reshape(x,(d,1)))
    # print("pj "+str(pj_x.shape))
    O,ux=outlyingness(x,loc,scale,u,pj_x)
    # coef=(cp.sign(pj_x[ux,:]-loc[ux])/scale[ux])
    # print("coef "+str(coef))
    if gd:
        coef2=cp.sign(pj_x[ux,:]-loc[ux])
        grad=coef2*u[ux,:]
    else:
        coef=(cp.sign(pj_x[ux,:]-loc[ux])/scale[ux])
        grad=coef*u[ux,:]
    # cupy.asnumpy(grad)
    return grad



# Gradient descent algorithm for computing the non-private median 
# step: step size 
# x: d x 1 numpy array for point at which depth is computed
# u: nvec x d numpy array of uniformly sampled unit vectors
# loc,scale: nvec x 1 numpy array of location and scale estimates for each u
def GD_update_PD(x,step,loc,scale,u,d):
    return x-step*grad_outlyingness(x,loc,scale,u,d)


# non private pd median
# IV: list of data projected onto the unit vectors (n x nvec), 
#            mu computed at all unit vectors (nvec x 1) and sigma (nvec x 1) computed at all unit vectors
# u: nvec x d matrix of unit vectors
# d: dimension
# step: step size 
# nstep: number of steps 
def non_private_pd(IV,u,d,step,nstep=1000,plot=False,x0=None,return_chain=False):
    ###### MCMC-lagevin dyanmics for private median
    if x0 is None:
        x0=mrng.standard_normal(size=(d))
        x0=cp.asarray(x0)
    x = []
    x.append(x0)
    for t in range(nstep):
        x.append(GD_update_PD(x[-1],step,IV[0],IV[1],u,d))
    if plot:
        x=cp.vstack(x)
        norm_x_p = cp.apply_along_axis(cp.linalg.norm, 1, x)
        plt.plot(np.arange(len(norm_x_p)),cp.asnumpy(norm_x_p))
        plt.ylabel('estimate norm', fontsize=18)
        plt.xlabel('time step', fontsize=18)
        plt.show()
    if return_chain:
        return x
    else:
        return x[-1]





######
### one step of langevin dynamics
## 
# x: d x 1 numpy array for point at which depth is computed
# step: step size for the langevin dynamics
# sqrt2step: just sqrt(2*step)
# d: integer dimension
# n: integer sample size
# u: nvec x d numpy array of uniformly sampled unit vectors
# eps: pure privacy parameter
# mrng: this is the seed for the mtalg package
def langevin_update_PD(x,u,loc,scale,n,d,eps,eta,tau,step,sqrt2step,mrng):
    # prior_grad=grad_prior(x,prior_mean,prior_cov_inv)
    # prior_grad=0
    o_grad=grad_outlyingness(x,loc,scale,u,d,False)
    return x-step*o_grad*eps/(4*eta)+sqrt2step*cp.asarray(mrng.standard_normal(size=d))


# IV: list of data projected onto the unit vectors (n x nvec), 
#            mu computed at all unit vectors (nvec x 1) and sigma (nvec x 1) computed at all unit vectors
# u: nvec x d matrix of unit vectors
# n: number of samples
# d: dimension
# tau: tau value for PTR
# eta: eta value for PTR
# eps: DP param 1
# step_l: step size for langevin dynamics
# nstep: number of steps for the langevin dynamics
# mrng: object from mtalg package for seed
                            
def run_exponential_mechanism(IV,u,n,d,eps,eta,tau,step_l,nstep=1000,mrng=None,plot=False,x0=None,return_chain=False):
    ###### MCMC-lagevin dyanmics for private median
    if x0 is None:
        x0=mrng.standard_normal(size=(d))
        x0=cp.asarray(x0)
    x = []
    x.append(x0)
    sqrt2step = cp.sqrt(2*step_l)
    for t in range(nstep):
        x.append(langevin_update_PD(x[-1],u,IV[0],IV[1],n,d,eps,eta,tau,step_l,sqrt2step,mrng))
    if plot:
        x=cp.vstack(x)
        norm_x_p = cp.apply_along_axis(cp.linalg.norm, 1, x)
        plt.plot(np.arange(len(norm_x_p)),cp.asnumpy(norm_x_p))
        plt.ylabel('estimate norm', fontsize=18)
        plt.xlabel('time step', fontsize=18)
        plt.show()
    if return_chain:
        return x
    else:
        return x[-1]


## These functions correspond to the testing algorithm
## Post editing with Dylan

## ## ## ## ## ## ## ## Volume check ## ## ## ## ## ## ## ## ## ## ## ## 

# This is computing the volume ratio for a given y, as in Lemma 5.3
# y: positive value to compute ratio at
# tau: tau value for PTR
# C1: \inf_u \hat \sigma u
# C2: \sup_u \hat \sigma u
# Delta_mu = \sup_u,v \hat \mu_u -\hat \mu_v
# eta: eta value for PTR
# eps: DP param 1
# delta: DP param 2
# d: dimension of data
def get_fv(y,tau,C1,C2,Delta_mu,eta,eps,delta,d):
    # print("y ",y)
    # print("over ",C1*(tau-y-2*eta)-Delta_mu)
    if((C1*(tau-y-2*eta)-Delta_mu)>0):
        numer=C2*(tau+2*eta)+Delta_mu
        denom=C1*(tau-y-2*eta)-Delta_mu
        ratio=numer/denom
        to_expon=d*cp.log(ratio)-eps*y/(4*eta)
        # print('Deltamu ',Delta_mu)
        # print("final ", to_expon)
    else:
        print("warnings -1")
        final_value=-1
    return to_expon


# This is doing the volume check at the minimum y 
# the minimum y is tau-2*eta-Delta_mu/C1-4*eta d/eps
# tau: tau value for PTR
# C1: \inf_u \hat \sigma u
# C2: \sup_u \hat \sigma u
# Delta_mu = \sup_u,v \hat \mu_u -\hat \mu_v
# eta: eta value for PTR
# eps: DP param 1
# delta: DP param 2
# d: dimension of data
def vol_check(tau,C1,C2,Delta_mu,eta,eps,delta,d):
    y_min=tau-2*eta-Delta_mu/C1-4*eta*d/eps
    return get_fv(y_min,tau,C1,C2,Delta_mu,eta,eps,delta,d) <= cp.log(delta)-eps/2



## ## ## ## ## ## ## ## Delta check ## ## ## ## ## ## ## ## ## ## ## ## 

# Compute Delta_k
# pj_data: data projected onto the unit vectors (n x nvec)
# fn_dict: dictionary of functions with keys mu, sigma, Smu, Ssigma and bhat
# k: k as in the paper (integer)
# tau: tau value for PTR
# C1: \inf_u \hat \sigma u
# def Delta(pj_data,fn_dict,k,tau,C1,n,scale,joint=False):
#     if joint:
#         vals=cp.apply_along_axis(fn_dict['joint'],axis=0,arr=pj_data,k=k,n=n)
#         # print(vals.shape)
#         S_mu=cp.max(vals[0,:])
#         S_sig=cp.max(vals[1,:])
#         # print(S_sig)
#         # if fn_dict['mad_flag']:
#         #     b_min=cp.min(scale-vals[2,:])
#             # print(b_min)
#         # else:
#         b_min=cp.min(vals[2,:])
#     else:
#         S_mu=cp.max(cp.apply_along_axis(fn_dict['Smu'],axis=0,arr=pj_data,k=k))
#         S_sig=cp.max(cp.apply_along_axis(fn_dict['Ssigma'],axis=0,arr=pj_data,k=k))
#         b_min=cp.min(cp.apply_along_axis(fn_dict['bhat'],axis=0,arr=pj_data,k=k))
#     # print(tau)
#     # print(b_min)
#     # print(S_mu)
#     # print(S_sig)
#     # print([cp.asnumpy(b_min),cp.asnumpy(C1)])
#     # print((tau*S_sig+S_mu)/np.min([cp.asnumpy(b_min),cp.asnumpy(C1)]))
#     return (tau*S_sig+S_mu)/np.min([cp.asnumpy(b_min),cp.asnumpy(C1)])

# Computes everything on a per unit vector basis. 
def Delta(pj_data,fn_dict,k,tau,scales,n):
    # Compute (tau*S_sig+S_mu)/np.min([cp.asnumpy(b_min),cp.asnumpy(C1)]) for fixed u instead
    vals=cp.apply_along_axis(fn_dict['joint'],axis=0,arr=pj_data,k=k,n=n)
    b_mins=cp.min(cp.column_stack((vals[2,:],scales)),axis=1)
        # print(vals.shape)
    deltas=(tau*vals[1,:]+vals[0,:])/b_mins
    delta=cp.max(deltas)
    return delta



## ## ## ## ## ## ## ## Lower bound on Safety margin ## ## ## ## ## ## ## ## ## ## ## ## 

# Finds the lower bound on the SM not through bs
# IV: list of data projected onto the unit vectors (n x nvec), 
#            mu computed at all unit vectors (nvec x 1) and sigma (nvec x 1) computed at all unit vectors
# fn_dict: dictionary of functions with keys mu, sigma, Smu, Ssigma and bhat
# eta: eta value for PTR
# tau: tau value for PTR
# eps: DP param 1
# delta: DP param 2
# d: dimension of data
# def find_SM_approx(IV,fn_dict,eta,tau,eps,delta,d,n,joint):
#     #Check the first condition
#     k=1
#     C1=cp.min(IV[1])
#     C2=cp.max(IV[1])
#     Delta_mu=cp.max(IV[0])-cp.min(IV[0])
#     cond_2  = vol_check(tau,C1,C2,Delta_mu,eta,eps,delta,d)
#     conditions_satis= cond_2
#     Delta_k = -1
#     while conditions_satis and k<12:
#         print(k)
#         k+=1
#         Delta_k =Delta(IV[2],fn_dict,k,tau,C1,n,joint)
#         # print("Delta k",Delta_k)
#         cond_1  = Delta_k<=eta
#         #we could replace eta with Delta_k!
#         conditions_satis=cond_1 and cond_2
#     print("Delta k",Delta_k)
#     print("Delta_k test ",cond_1)
#     print("Volume test ",cond_2)
#     sm=k-2
#     return sm



# Finds the lower bound on the SM via binary search
# IV: list of data projected onto the unit vectors (n x nvec), 
#            mu computed at all unit vectors (nvec x 1) and sigma (nvec x 1) computed at all unit vectors
# fn_dict: dictionary of functions with keys mu, sigma, Smu, Ssigma and bhat
# eta: eta value for PTR
# tau: tau value for PTR
# eps: DP param 1
# delta: DP param 2
# # d: dimension of data
# def find_SM_approx(IV,fn_dict,eta,tau,eps,delta,d,n,joint,curr_max=22):
#     #prelimm
#     C1=cp.min(IV[1])
#     C2=cp.max(IV[1])
#     Delta_mu=cp.max(IV[0])-cp.min(IV[0])
#     #check for k=1
#     curr_min=1
#     Delta_k =Delta(IV[2],fn_dict,curr_min,tau,C1,n,joint)
#     cond_1  = Delta_k<=eta
#     #volume condition
#     cond_2  = vol_check(tau,C1,C2,Delta_mu,eta,eps,delta,d)
#     conditions_satis=cond_1 and cond_2
#     # print(1)
#     if conditions_satis:
#         while curr_max>curr_min:
#             Delta_k =Delta(IV[2],fn_dict,curr_max,tau,C1,n,joint)
#             conditions_satis=Delta_k<=eta
#             # print(curr_max)
#             if not conditions_satis:
#                 curr_max=int(curr_max/2)
#             else:
#                 curr_min=curr_max
#     print("Delta k",Delta_k)
#     print("Delta_k test ",cond_1)
#     print("Volume test ",cond_2)
#     print("SM ",curr_min-1)
#     sm=curr_min-1
#     return sm


# Finds the lower bound on the SM via binary search, then one at a time search
# IV: list of data projected onto the unit vectors (n x nvec), 
#            mu computed at all unit vectors (nvec x 1) and sigma (nvec x 1) computed at all unit vectors
# fn_dict: dictionary of functions with keys mu, sigma, Smu, Ssigma and bhat
# eta: eta value for PTR
# tau: tau value for PTR
# eps: DP param 1
# delta: DP param 2
# d: dimension of data
def find_SM_approx(IV,fn_dict,eta,tau,eps,delta,d,n,joint,curr_max=32):
    #prelimm
    C1=cp.min(IV[1])
    C2=cp.max(IV[1])
    Delta_mu=cp.max(IV[0])-cp.min(IV[0])
    #check for k=1
    curr_min=1
    # Delta_k =Delta(IV[2],fn_dict,curr_min,tau,C1,n,IV[1],joint)
    Delta_k = Delta(IV[2],fn_dict,curr_min,tau,IV[1],n)
    cond_1  = Delta_k<=eta
    #volume condition
    cond_2  = vol_check(tau,C1,C2,Delta_mu,eta,eps,delta,d)
    conditions_satis=cond_1 and cond_2
    # print(1)
    if conditions_satis:
        while curr_max>curr_min:
            # Delta_k =Delta(IV[2],fn_dict,curr_max,tau,C1,n,IV[1],joint)
            Delta_k = Delta(IV[2],fn_dict,curr_max,tau,IV[1],n)
            cond_1=Delta_k<=eta
            conditions_satis=cond_1
            # print(curr_max)
            if not conditions_satis:
                curr_max=int(curr_max/2)
            else:
                curr_min=curr_max
    while conditions_satis and curr_min<10:
        curr_min+=1
        # Delta_k =Delta(IV[2],fn_dict,curr_min,tau,C1,n,IV[1],joint)
        Delta_k = Delta(IV[2],fn_dict,curr_min,tau,IV[1],n)
        cond_1  = Delta_k<=eta
        conditions_satis=cond_1 and curr_min<10
    if not conditions_satis or curr_min>10:
        curr_min+=1
    # print("Delta k",Delta_k)
    # print("Delta_k test ",cond_1)
    # print("Volume test ",cond_2)
    # print("SM ",curr_min-2)
    sm=curr_min-2
    return [sm,Delta_k,cond_1,cond_2]



## ## ## ## ## ## ## ## Test function ## ## ## ## ## ## ## ## ## ## ## ## 


# Executes test in PTR
# IV: list of data projected onto the unit vectors (n x nvec), 
#            mu computed at all unit vectors (nvec x 1) and sigma (nvec x 1) computed at all unit vectors
# fn_dict: dictionary of functions with keys mu, sigma, Smu, Ssigma and bhat
# eta: eta value for PTR
# tau: tau value for PTR
# delta: DP param 2
# eps: DP param 1
# d: dimension of data
# n: sample size
# joint is whether or not to compute jointly, its fater so usually you want this
def test(IV,fn_dict,eta,tau,delta,eps,d,n,joint):
    details=find_SM_approx(IV,fn_dict,eta,tau,eps,delta,d,n,joint)
    sm=details[0]
    # print("SM ",sm)
    W=cp.random.laplace(loc=0, scale=1, size=1)
    noisy_sm=sm+2*W/eps
    RHS=2*cp.log(1/(2*delta))/eps
    test=noisy_sm>RHS
    if not test:
        print("Delta k",details[1])
        print("Delta_k test ",details[2])
        print("Volume test ",details[3])
        print("SM ",sm)
    return test







#this is an old implementation we might want to compare to

# def find_SM_approx(IV,eta,tau,delta,d):
#     #Check the first condition
#     k=0
#     C1=cp.min(IV[1])
#     C2=cp.max(IV[1])
#     Delta_mu=cp.max(IV[0])-cp.min(IV[0])
#     conditions_satis=True
#     while conditions_satis:
#         k+=1
#         Delta_k =Delta(IV[2],k,tau,C1)
#         # print("Delta k",Delta_k)
#         cond_1  = Delta_k<=eta
#         #we could replace eta with Delta_k!
#         cond_2  = vol_check(tau,C1,C2,Delta_mu,eta,eps,delta,d)
#         conditions_satis=cond_1 and cond_2
#     print("Delta k",Delta_k)
#     print("test 1 ",cond_1)
#     print("test 2 ",cond_2)
#     sm=k-1
#     return sm




# This is an old version I might want to compare to 
# def vol_check(tau,C1,C2,Delta_mu,eta,eps,delta,d):
#     a=C1*tau-2*C1*eta-Delta_mu
#     b=eps/(4*eta)
#     x=(b*a-d*C1)/(b*C1)
#     return get_fv(x,tau,C1,C2,Delta_mu,eta,eps,delta,d) <= cp.log(delta)





#In this cell are the functions for executing the test portion of the mechanism for specific sigma and mu

#Its the S and the b function for a single u , then we just apply these along axis to all u 
#assumes x is sorted... 
# def S_med_mad_b_joint(x,k,n):
#     k=int(k)
#     a=int(np.floor((n+1)/2))
#     # minus 1 for 0 indexing
#     S_med = cp.min(cp.array([x[a+k-1] - x[a-1], x[a-1] - x[a-k-1]]))
#     q_prime=1/4-2*k/n-0.001
#     low=int(np.floor(q_prime*n))-1
#     up=int(np.floor((1-q_prime)*n))-k
#     spacing=cp.max(x[(low+k):(up+k)]- x[low:up] )
#     # cur_max=-1
#     # for j in range(low,up):
#     #     # minus 1 for 0 indexing
#     #     spacing=x[j+k-1]-x[j-1]
#     #     if spacing>cur_max:
#     #         cur_max=spacing
#     S_mad=2*spacing
#     b_min=S_mad
#     return cp.array([S_med,S_mad ,b_min])


#Its the S and the b function for a single u , then we just apply these along axis to all u 
# assumes x is sorted... 
def S_med_iqr_b_joint(x,k,n):
    k=int(k)
    a=int(cp.floor((n+1)/2))
    b=int(cp.floor(n/4))
    c=int(cp.floor(3*n/4))
    # minus 1 for 0 indexing
    S_med = cp.min(cp.array([x[a+k-1] - x[a-1], x[a-1] - x[a-k-1]]))
    IQR=x[c-1] - x[b-1]
    smaller_iqr = x[c-k-1]-x[b+k-1]
    larger_iqr =  x[c+k-1]-x[b-k-1]
    b_min=cp.min(cp.array([smaller_iqr,IQR]))
    S_iqr=cp.max(cp.array([larger_iqr-IQR,IQR-smaller_iqr]))
    return cp.array([S_med,S_iqr ,b_min])

# computing things (Ss, and b hat) jointly so its faster
# assumes x is sorted... 
def S_med_mad_b_joint(x,k,n):
    k=int(k)
    a=int(cp.floor((n+1)/2))
    # minus 1 for 0 indexing
    S_med = cp.min(cp.array([x[a+k-1] - x[a-1], x[a-1] - x[a-k-1]]))
    b_vals=cp.array([])
    S_vals=cp.array([])
    low=a-k-1
    up=a+k
    for j in range(low,up):
        # zz=order_stat(x,j)
        zz=x[j]
        ls,bv=joint_g_LS_z(zz,x,k,n)
        S_vals=cp.append(S_vals,ls)
        b_vals=cp.append(b_vals,bv)
        # S_vals=cp.append(S_vals,LS_z(zz,x,k,n))
        # b_vals=cp.append(b_vals,g_z(zz,x,n))
    b_min=cp.min(b_vals)
    S_mad=cp.max(S_vals)
    return cp.array([S_med,S_mad ,b_min])




# Its the S and the b function for a single u , then we just apply these along axis to all u 
# assumes x is sorted!!
def S_hb_mad_b_joint(x,k,n,beta=2):
    k=int(k)
    a=int(cp.floor((n+1)/2))
    q_prime=1/4-2*k/n-0.001
    low=int(np.floor(q_prime*n))-1
    up=int(np.floor((1-q_prime)*n))-k
    # cur_max=cp.max(x[(low+k):(up+k)]- x[low:up] )
    #This is not bmin, have to subtract this from the mad of x
    # b_min=2*cur_max
    # S_mad=2*cur_max
    b_vals=cp.array([])
    S_vals=cp.array([])
    for j in range(a-k-1,a+k):
        zz=x[j]
        ls,bv=joint_g_LS_z(zz,x,k,n)
        S_vals=cp.append(S_vals,ls)
        b_vals=cp.append(b_vals,bv)
    b_min=cp.min(b_vals)
    S_mad=cp.max(S_vals)
    #huber
    condition=((x[a+k-1] -x[a-k-1])/b_min)<=beta
    if condition:
        S_hb=2*float(k)**2*beta/float(n)
    else:
        S_hb=cp.inf
    # print(type(S_hb))
    # print(type(S_mad))
    # print(type(b_min))
    return cp.array([S_hb,S_mad.get() ,b_min.get()])

# computing g and local sens things jointly so its faster

def joint_g_LS_z(z,x,k,n):
    y=cp.abs(x-z)
    ls=S_med(y,k,n)
    g=order_stat(y, int((n+1)/2))
    return [ls,g]

# def g_z(z,x,n):
#     y=cp.abs(x-z)
#     g=order_stat(y, int((n+1)/2))
#     return g



    
    
## add S and b hat for mad
# order stat wrapper
def order_stat(x,k):
    return cp.partition(x, k-1)[k-1]

# def LS_z(z,x,k,n):
#     y=cp.abs(x-z)
#     ls=S_med(y,k,n)
#     return ls

# def g_z(z,x,n):
#     y=cp.abs(x-z)
#     g=order_stat(y, int((n+1)/2))
#     return g

# b hat for mad
def b_hat_mad(x,k,n):
    q_prime=1/4-2*k/n-0.000001
    low=int(np.floor(q_prime*n))
    up=int(np.floor((1-q_prime)*n))-k
    cur_max=-1
    for j in range(low,up):
        # minus 1 for 0 indexing
        spacing=x[j+k-1]-x[j-1]
        if spacing>cur_max:
            cur_max=spacing
    return mad(x)-2*cur_max


# def b_hat_mad(x,k):
#     # x=cp.sort(x)
#     n=len(x)
#     #subtracting 1 for 0 indexing
#     # bk=x[cp.floor((n+1)/2)+k-1]
#     # ak=x[cp.floor((n+1)/2)-k-1]
#     y=cp.array([])
#     low=int(cp.floor((n+1)/2)-k-1)
#     up=int(cp.floor((n+1)/2)+k)
#     for j in range(low,up):
#         zz=order_stat(x,j)
#         y=cp.append(y,g_z(zz,x))
#     b_min=cp.min(y)
#     return b_min

## S for mad
def S_mad(x,k,n):
    q_prime=1/4-2*k/n-0.000001
    low=int(np.floor(q_prime*n))
    up=int(np.floor((1-q_prime)*n))-k
    cur_max=-1
    for j in range(low,up):
        # minus 1 for 0 indexing
        spacing=x[j+k-1]-x[j-1]
        if spacing>cur_max:
            cur_max=spacing
    return 2*cur_max

# def S_mad(x,k):
#     # x=cp.sort(x)
#     n=len(x)
#     #subtracting 1 for 0 indexing
#     # bk=x[cp.floor((n+1)/2)+k-1]
#     # ak=x[cp.floor((n+1)/2)-k-1]
#     # S_med=bk-ak
#     y=cp.array([])
#     low=int(cp.floor((n+1)/2)-k-1)
#     up=int(cp.floor((n+1)/2)+k)
#     for j in range(low,up):
#         zz=order_stat(x,j)
#         y=cp.append(y,LS_z(zz,x,k))
#     S_mad=cp.max(y)
#     return S_mad    
    
    
    

## S for median
def S_med(x,k,n):
    q=int((n+1)/2)
    # q3 = cp.percentile(x, q/n+100*k/n)
    # q2 = cp.percentile(x, q/n)
    # q1 = cp.percentile(x, q/n-100*k/n)
    q3 = order_stat(x, q+k)
    q2 = order_stat(x, q)
    q1 = order_stat(x, q-k)
    ls = cp.min(cp.array([q3 - q2, q2 - q1]))
    # ls = np.min(np.array([q3 - q2, q2 - q1]))
    return ls













# \hat b (or b_-) for the IQR
def b_hat_iqr(x,k):
    n=len(x)
    q1=100*cp.floor(n/4)
    q2=100*cp.floor(3*n/4)
    IQR=iqr(x)
    smaller_iqr = (cp.percentile(x, q2/n-100*k/n)-cp.percentile(x, q1/n+100*k/n))
    return cp.min(cp.array([smaller_iqr,IQR]))
## S for iqr
def S_iqr(x,k):
    n=len(x)
    q1=100*cp.floor(n/4)
    q2=100*cp.floor(3*n/4)
    IQR=iqr(x)
    larger_iqr = cp.percentile(x, q2/n+100*k/n)-cp.percentile(x, q1/n-100*k/n) - IQR
    smaller_iqr = IQR-(cp.percentile(x, q2/n-100*k/n)-cp.percentile(x, q1/n+100*k/n))
    ls = cp.max(cp.array([larger_iqr,smaller_iqr]))
    return ls





## S for Huber location paired with MAD
def S_Huber(x,k,beta=2.0):
    n=len(x)
    q=100*cp.floor((n+1)/2)
    q3 = cp.percentile(x, q/n+100*k/n)
    q1 = cp.percentile(x, q/n-100*k/n)
    condition=(q3 -q1)/b_hat_mad(x,k)<=beta
    if condition:
        return 2*float(k)**2*beta/float(n)
    else:
        return cp.inf

##Putting it all together 

# data: data set, n x d
# fn_dict: dictionary of functions with keys mu, sigma, Smu, Ssigma and bhat
# tau: tau value for PTR
# eta: eta value for PTR
# eps: DP param 1
# delta: DP param 2
# nvec: num unit vectors N
# mrng: object from mtalg package for seed (not needed now)
# step_l: step size
# nstep: num steps in LD
# u: nvec x d numpy array of uniformly sampled unit vectors
# x0: starting val for chain
# return_chain: return the MC?
# IV: we may pass common IV to avoid repeat computation
# joint: computing some parts of median and mad jointly to improve computation
def compute_PTR_PD_Median(data,fn_dict,tau,eta,eps,delta,nvec=100,mrng=None,step_l=0.0000002,nstep=1000,u=None,x0=None,return_chain=False,IV=None,joint=True):
    n=data.shape[0]
    d=data.shape[1]
    #compute unit vectors
    if u is None:
        u=generate_UV(nvec,d,mrng)
    #compute values of mu and sigma for all u 
    if IV is None:
        IV=get_initial_values(u,data,fn_dict['mu'],fn_dict['sigma'])
    test_passed=test(IV,fn_dict,eta,tau,delta,eps,d,n,joint=joint)
    if test_passed:
        return run_exponential_mechanism(IV,u,n,d,eps,eta,tau,step_l=step_l,nstep=nstep,mrng=mrng,x0=x0,return_chain=return_chain)
    else:
        return "Test failed"

