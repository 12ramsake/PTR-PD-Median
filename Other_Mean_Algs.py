# coding: utf-8
import torch
import argparse
import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt
import diffprivlib as dpl


'''
extra functions
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_budget', default=.5, type=float, help='Total privacy budget')
    parser.add_argument('--d', default=10, type=int, help='Feature dimension (dimension of synthetic data)')
    parser.add_argument('--n', default=3000, type=int, help='Number of samples to synthesize (for synthetic data)')
    parser.add_argument('--u', default=33, type=float, help='Initial upper bound for covariance')
    
    parser.add_argument('--fig_title', default=None, type=str, help='figure title')
    parser.add_argument('-f', default=None, type=str, help='needed for ipython starting')
    
    opt = parser.parse_args()
    return opt

def cov_nocenter(X):
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

def cov(X):
    X = X - X.mean(0)
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

'''
PSD projection
'''
def psd_proj_symm(S):
    U, D, V_t = torch.svd(S)
    D = torch.clamp(D, min=0, max=None).diag()
    A = torch.mm(torch.mm(U, D), U.t()) 
    return A

'''
Mean Estimation Methods --------------------------------------------------------
'''

'''
Fine mean estimation algorithm 
 - list params are purely for graphing purposes and can be ignored if not needed
returns: fine DP estimate for mean
'''
def fineMeanEst(x, sigma, R, epsilon, epsilons=[], sensList=[], rounding_outliers=False):
    B = R+sigma*3
    sens = 2*B/(len(x)*epsilon) 
    epsilons.append([epsilon])
    sensList.append([sens])
    if rounding_outliers:
        for i in x:
            if i > B:
                i = B
            elif i < -1*B:
                i =  -1*B
    noise = np.random.laplace(loc=0.0, scale=sens)
    result = sum(x)/len(x) + noise 
    return result

'''
Coarse mean estimation algorithm with Private Histogram
returns: [start of intrvl, end of intrvl, freq or probability], bin number
- the coarse mean estimation would just be the midpoint of the intrvl (in case this is needed)
'''
def privateRangeEst(x, epsilon, delta, alpha, R, sd):
    # note alpha ∈ (0, 1/2)
    r = int(math.ceil(R/sd))
    bins = {}
    for i in range(-1*r,r+1):
        start = (i - 0.5)*sd # each bin is s ((j − 0.5)σ,(j + 0.5)σ]
        end = (i + 0.5)*sd 
        bins[i] = [start, end, 0] # first 2 elements specify intrvl, third element is freq
    # note: epsilon, delta ∈ (0, 1/n) based on https://arxiv.org/pdf/1711.03908.pdf Lemma 2.3
    # note n = len(x)
    # set delta here
    L = privateHistLearner(x, bins, epsilon, delta, r, sd)
    return bins[L], L


# helper function
# returns: max probability bin number
def privateHistLearner(x, bins, epsilon, delta, r, sd): # r, sd added to transmit info
    # fill bins
    max_prob = 0
    max_r = 0

    # creating probability bins
    for i in x:
        r_temp = int(round(i/sd))
        if r_temp in bins:
            bins[r_temp][2] += 1/len(x)
        
    for r_temp in bins:
        noise = np.random.laplace(loc=0.0, scale=2/(epsilon*len(x)))
        if delta == 0 or r_temp < 2/delta:
            # epsilon DP case
            bins[r_temp][2] += noise
        else:
            # epsilon-delta DP case
            if bins[r_temp][2] > 0:
                bins[r_temp][2] += noise
                t = 2*math.log(2/delta)/(epsilon*len(x)) + (1/len(x))
                if bins[r_temp][2] < t:
                    bins[r_temp][2] = 0
        
        if bins[r_temp][2] > max_prob:
            max_prob = bins[r_temp][2]
            max_r = r_temp
    return max_r


'''
Two shot algorithm
- may want to optimize distribution ratio between fine & coarse estimation

eps1 = epsilon for private histogram algorithm
eps2 = epsilon for fine mean estimation algorithm

returns: DP estimate for mean
'''
def twoShot(x, eps1, eps2, delta, R, sd):
    alpha = 0.5
    # coarse estimation
    [start, end, prob], r = privateRangeEst(x, eps1, delta, alpha, R, sd)
    for i in range(len(x)):
        if x[i] < start - 3*sd:
            x[i] = start - 3*sd
        elif x[i] > end + 3*sd:
            x[i] = end + 3*sd
    # fine estimation with smaller range (less sensitivity)
    est = fineMeanEst(x, sd, 3.5*sd, eps2)
    return est


'''
Privately estimating covariance.
'''




def cov_est_step(X, A, rho, cur_iter, args,beta):
    """
    One step of multivariate covariance estimation, scale cov a.
    """
    # print(X.shape)
    if len(X.shape)>2:
        # print(X.shape)
        X=torch.squeeze(X)
        # print('after' ,X.shape)
    n, d = X.shape

    #Hyperparameters
    gamma = gaussian_tailbound(d, beta)
    # Was here before
    # gamma = gaussian_tailbound(d, 0.1)
    # eta = 0.5*(2*(np.sqrt(d/n)) + (np.sqrt(d/n))**2)
    eta = 2*(np.sqrt(d/n) + np.sqrt(2*np.log(2/beta)/n)) + (np.sqrt(d/n) + np.sqrt(2*np.log(2/beta)/n))**2
    nu = (gamma**2 / (n*np.sqrt(rho))) * (2*np.sqrt(d) + 2*d**(1/16) * np.log(d)**(1/3) + (6*(1 + (np.log(d)/d)**(1/3))*np.sqrt(np.log(d)))/(np.sqrt(np.log(1 + (np.log(d)/d)**(1/3)))) + 2*np.sqrt(2*np.log(1/beta)))

    
    #truncate points
    # W = torch.mm(X, A)
    W = torch.mm(X, A.t())
    W_norm = np.sqrt((W**2).sum(-1, keepdim=True))
    norm_ratio = gamma / W_norm
    large_norm_mask = (norm_ratio < 1).squeeze()
    
    W[large_norm_mask] = W[large_norm_mask] * norm_ratio[large_norm_mask]
    
    # noise
    Y = torch.randn(d, d)
    noise_var = (gamma**4/(rho*n**2))
    Y *= np.sqrt(noise_var)    
    #can also do Y = torch.triu(Y, diagonal=1) + torch.triu(Y).t()
    Y = torch.triu(Y)
    Y = Y + Y.t() - Y.diagonal().diag_embed() # Don't duplicate diagonal entries
    Z = (torch.mm(W.t(), W))/n
    #add noise    
    Z = Z + Y
    #ensure psd of Z
    Z = psd_proj_symm(Z)
    
    U = Z + (nu+eta)*torch.eye(d)
    inv = torch.inverse(U)
    inv_sqrt=compute_sqrt_mat(inv)
    # invU, invD, invV = inv.svd()
    # inv_sqrt = torch.mm(invU, torch.mm(invD.sqrt().diag_embed(), invV.t()))
    A = torch.mm(inv_sqrt, A)
    return A, Z



def compute_sqrt_mat(A):
    U, D, V = A.svd()
    inv_sqrt = torch.mm(U, torch.mm(D.sqrt().diag_embed(), V.t()))
    return inv_sqrt


def cov_est(X, args,beta=0.1):
    """
    Multivariate covariance estimation.
    Returns: zCDP estimate of cov.
    """
    A = torch.eye(args.d) / np.sqrt(args.u)
    assert len(args.rho) == args.t
    
    for i in range(args.t-1):
        A, Z = cov_est_step(X, A, args.rho[i], i, args,beta/(4*(args.t-1)))
    A_t, Z_t = cov_est_step(X, A, args.rho[-1], args.t-1, args,beta/4)
    
    cov = torch.mm(torch.mm(A.inverse(), Z_t), A.inverse().t())
    return cov






def gaussian_tailbound(d,b):
    return ( d + 2*( d * np.log(1/b) )**0.5 + 2*np.log(1/b) )**0.5

def mahalanobis_dist(M, Sigma):
    Sigma_inv = torch.inverse(Sigma)
    U_inv, D_inv, V_inv = Sigma_inv.svd()
    Sigma_inv_sqrt = torch.mm(U_inv, torch.mm(D_inv.sqrt().diag_embed(), V_inv.t()))
    M_normalized = torch.mm(Sigma_inv_sqrt, torch.mm(M, Sigma_inv_sqrt))
    return torch.norm(M_normalized - torch.eye(M.size()[0]), 'fro')

''' 
Functions for mean estimation
'''

##    X = dataset
##    c,r = prior knowledge that mean is in B2(c,r)
##    t = number of iterations
##    Ps = 
def multivariate_mean_iterative(X, c, r, t, Ps,beta=0.1):
    for i in range(t-2):
        c, r = multivariate_mean_step(X, c, r, Ps[i],beta/(4*(t-1)))
    c, r = multivariate_mean_step(X, c, r, Ps[t-1],beta/4)
    return c

def multivariate_mean_step(X, c, r, p,beta):
    n, d = X.shape

    ## Determine a good clipping threshold
    gamma = gaussian_tailbound(d,beta)
    clip_thresh = min((r**2 + 2*r*3 + gamma**2)**0.5 , r + gamma) #3 in place of sqrt(log(2/beta))
        
    ## Round each of X1,...,Xn to the nearest point in the ball B2(c,clip_thresh)
    x = X - c
    mag_x = torch.linalg.norm(x, axis=1)
    outside_ball = (mag_x > clip_thresh)
    x_hat = (x.T / mag_x).T
    if torch.sum(outside_ball)>0:
        X[outside_ball] = c.float() + (x_hat[outside_ball].float() * clip_thresh)
    
    ## Compute sensitivity
    delta = 2*clip_thresh/float(n)
    # print(delta)
    # print(p)
    sd = delta/(2*p)**0.5
    
    ## Add noise calibrated to sensitivity
    Y = np.random.normal(0, sd, size=d)
    c = torch.sum(X, axis=0)/float(n) + Y
    r = ( 1/float(n) + sd**2 )**0.5 * gaussian_tailbound(d,0.01)
    return c, r

def L1(est): # assuming 0 vector is gt
    return np.sum(np.abs(est))
    
def L2(est): # assuming 0 vector is gt
    return np.linalg.norm(est)

# Estimates mean when cov unknown
def overall_mean(X,args):
    # n, d = X.shape
    # print('before', X.shape)
    if len(X.shape)>2:
        # print(X.shape)
        X=torch.squeeze(X)
    if len(X.shape)<2:
        X=X.view(X.shape[0], 1)
    row_diff = torch.diff(X, axis=0)[::2]
    
    Sigma=cov_est(row_diff,args)/2

    # U, S, Vh =torch.linalg.svd(Sigma)
    # D = torch.diag(torch.sqrt(S))
    # sqrt_mat = U @ D @ Vh
    sqrt_mat = compute_sqrt_mat(Sigma)
    adj = torch.linalg.inv(sqrt_mat)

    # print('after', X.shape)
    whitened=X @ adj
    # args.t=3
    # args.r=10*np.sqrt(args.d)
    # args.Ps= torch.tensor([1/3.0, 1/2.0, 1.0])
    # multivariate_mean_iterative(X, c, r, t, Ps)
    mean_est = multivariate_mean_iterative(whitened,args.c,args.r,args.t,args.Ps)
    mean_est = mean_est.float() @ sqrt_mat
    return [mean_est, Sigma]


# Wrapper for above 
def COINPRESS(X,n,d,rho,c,r,u):
    args = parse_args()
    args.d = d
    args.n = n
    nm=torch.tensor([1/3.0, 1/2.0, 1.0])
    args.rho= rho*(nm/nm.sum())/2
    args.t=3
    args.c=c
    args.r=r
    args.u=u
    # args.Ps= [1.0/3.0, 1.0/2.0, 1.0]
    args.Ps= rho*(nm/nm.sum())/2
    return overall_mean(X,args)



## PRIME mean algorithm (not used in PD manuscript - this code is taken from their Git, and sometimes it had to be debugged so there may be minor changes)
def M(data, n):
    centered_X = data-data.mean(0)
    return 1/n*centered_X.T@centered_X

def output_perturbation(sigma,d):
    vec = np.random.normal(0, sigma, d**2).reshape(d,d)
    
    iu = np.triu_indices(d,1)
    il = (iu[1],iu[0])
    vec[il]=vec[iu]
    return vec


def filter_1d(tail_indices, tau, epsilon, delta, B, d):
    n = tau.shape[0]
    psi = np.sum(tau[tail_indices]-1)/n+np.random.laplace(0,B**2*d/(n*epsilon))
    bins = np.geomspace(1/4, B**2*d, num=2+int(np.log(B**2*d)))
    hist,  _ = np.histogram(tau[tail_indices], bins=bins)
    hist = hist/n
    
    for h in range(len(hist)):
        if hist[h]!=0:
            hist[h] += np.random.laplace(0, 2/(n*epsilon))
        if hist[h]<2*np.log(1/delta)/(epsilon*n)+1/n:
            hist[h] = 0

    for l in range(len(bins)-2, -1,-1):

        if np.sum((bins[l+1:]-bins[l])*hist[l:])>=0.3*psi:
            return bins[l]
    
    return None

        
def dp_range(X, epsilon,delta, R):
    n = X.shape[0]
    d = X.shape[1]
    x_bar = np.zeros(d)
    for i in range(d):
        bins = np.linspace(-R-0.5, R+0.5, 2*int(R))
        hist,  _ = np.histogram(X[:,i], bins=bins)
        hist = hist/n
        for h in range(len(hist)):
            if hist[h]!=0:
                hist[h] += np.random.laplace(0, 2/(n*epsilon/d))
            if hist[h]<2*np.log(d/delta)/(epsilon*n/d)+1/n:
                hist[h] = 0
        x_bar[i] = bins[np.argmax(hist)]+bins[np.argmax(hist)+1]
                
    return x_bar, 4*np.sqrt(np.log(d*n/0.9))


def filter_MMW(X, alpha, T1, T2, epsilon,delta, C, B):
    n = X.shape[0]
    d = X.shape[1]
    epsilon_1 = epsilon/(4*T1)
    epsilon_2 = epsilon/(4*T1*T2)
    delta_1 = delta/(4*T1)
    delta_2 = delta/(4*T1*T2)
    
    S = np.arange(0,n)
    
    for _ in range(T1):
        lamb_s = np.linalg.norm(M(X[S], n)-np.identity(d), 2)+np.random.laplace(0,2*B**2*d/(n*epsilon_1))
        output = np.mean(X[S], axis=0)+np.random.normal(0, 2*B*np.sqrt(2*d*np.log(1.25/delta_1))/(n*epsilon_1), d )
        if len(S)<0.55*n:
            # print('failed')
            return None
        if lamb_s<C*alpha*np.log(1/alpha):
            # print('succeded, '+str(S.max())+str("  ")+str(np.linalg.norm(output))+'  '+str(np.sum(S>=n-int(alpha*n))))
            return output
        alpha_s = 1/(10*(0.01/C+1.01)*lamb_s)

        Sigma_list = []
        for _ in range(T2):

            lamb_t = np.linalg.norm(M(X[S], n)-np.identity(d), 2)+np.random.laplace(0,2*B**2*d/(n*epsilon_2))
            if lamb_t<lamb_s*0.5:
                # print('next epoch')
                break
            else:
                Sigma = M(X[S],n)+output_perturbation(2*B*B*d*np.sqrt(2*np.log(1/delta_2))/(n*epsilon_2),d)
                Sigma_list.append(Sigma)
                sum_sigma = alpha_s*(np.array(Sigma_list)-np.identity(d)).sum(0)
                U = np.exp(sum_sigma)/np.trace(np.exp(sum_sigma))

                psi = np.trace((M(X[S], n)-np.identity(d))@(U.T))+np.random.laplace(0,2*B**2*d/(n*epsilon_2))
                if psi <= lamb_t/5.5:
                    continue
                else:
                    mu_t = np.mean(X[S], axis=0)+np.random.normal(0, 2*B*np.sqrt(2*d*np.log(1/delta_2))/(n*epsilon_2), d )
                    tau = (((X-mu_t)@U)*(X-mu_t)).sum(1)
                    sorted_tau_thres = np.sort(tau[S])[len(S)-2*int(alpha*n)]
                    
                    tail_indices = []
                    for l in S:
                        if tau[l]>=sorted_tau_thres:
                            tail_indices.append(l)
                    rho = filter_1d(tail_indices, tau, epsilon=epsilon_2, delta=delta_2, B=B, d=d)

                    S_remove = []
                    good = []
                    bad = []
                    for ind in tail_indices:
                        if tau[ind]>= np.random.uniform(0,1)*rho:
                            if ind>=n-int(alpha*n):
                                bad.append(ind)
                            else:
                                good.append(ind)
                            S_remove.append(ind)
                    # plt.figure()
                    # plt.hist(tau[bad], label='bad',bins=100)
                    # plt.hist(tau[good], label='good',bins=100)
                    # plt.legend()
                    # plt.show()

                    S = np.setdiff1d(S, np.array(S_remove))
          
    return output



def PRIME(epsilon, delta, X, alpha, R,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)  
    n = X.shape[0]
    d = X.shape[1]
    
    x_bar, B = dp_range(X, 0.01*epsilon,0.01*delta, R=R)

    for i in range(d):
        X[:,i] = np.clip(X[:,i], a_min = x_bar[i]-B, a_max = x_bar[i]+B) 

    C = 2
    if d>1:
        T1 = int(np.log(B*np.sqrt(d)))
        T2 = int(np.log(d))
        if T2==0:
            T2=2
    else:
        T1 = 2
        T2 = 2

    mean = filter_MMW(X=X, alpha=alpha, T1=T1, T2=T2, epsilon=0.99*epsilon,delta=0.99*delta, C=C, B=B)
        
    return mean
    
# This is the regular histogram based DP mean
def DPmean(epsilon, delta, X, alpha, R):
    n = X.shape[0]
    d = X.shape[1]
    
    x_bar, B = dp_range(X, epsilon,delta, R=R)
    

    S = np.arange(0,n)
    S_bad = []
    for i in range(n):
        for j in range(d):
            if X[i][j]>=x_bar[j]+B or X[i][j]<=x_bar[j]-B:
                
                S_bad.append(i)
    S = np.setdiff1d(S, np.array(S_bad))
        
    
    return np.mean(X[S], axis=0)
    



# heavy tailed mean estimation
def PDPRE(data,alpha,R,eps,k=2):
    # compute r and the number of bins - here alpha is error and k is the number of moments
    r=10/(alpha**(1/(k-1)))
    bins= np.arange(-R - 2*r, R + 2*r + 1, 2*r)
    # Private DP histogram
    counts, bin_edges = dpl.tools.histogram(data, eps,
                                             bins=bins, range=(-R - 2*r,R + 2*r))
    index_of_max = np.argmax(counts)
    I=np.array([bin_edges[index_of_max]-2*r,2*r+bin_edges[index_of_max+1]])
    return I

def DPMean(data,alpha,R,eps,beta=0.1):
    m=int(5*np.log(2/beta))
    n=len(data)
    j=int(n/2)
    k=int(j/m)
    # print(k)
    Z=data[0:(j+1)]
    W=data[(j+1):]
    r=10/alpha
    bins= np.arange(-R - 2*r, R + 2*r + 1, 2*r)
    means=np.zeros(m)
    for i in range(m):
        # print(int(i*k),int((i+1)*k-1))
        a=int(i*k)
        b=int((i+1)*k-1)
        Zi=Z[a:b]
        # print(Zi)
        # Private DP histogram
        counts, bin_edges = dpl.tools.histogram(Zi, eps,
                                                bins=bins, range=(-R - 2*r,R + 2*r))
        index_of_max = np.argmax(counts)
        interval=np.array([bin_edges[index_of_max]-2*r,2*r+bin_edges[index_of_max+1]])
        # interval=PDPRE(Zi,alpha,R,eps/2)
        ri=interval[1]-interval[0]
        Yi1=W[a:b]
        # print(Yi1)
        Yi=np.clip(Yi1,interval[0],interval[1])
        # print(np.mean(Yi))
        means[i]=np.mean(Yi)+np.random.laplace(2/eps,size=1)*ri/k
    return np.median(means)


    


### Duchi 2014 clipped mean
def duchi_2014(data, a,b, rho,r=np.nan,k=2,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)   
    if r is np.nan:
        r=((b-a)/4)    
    n = len(data)
    ub = r*((n**2)*rho)**(1/(2*k))
    lb = -ub
    Z=np.clip(data,lb,ub)
    np_mean=np.mean(Z)
    V=np.random.normal(scale=1/ np.sqrt(2*rho) , size=1)
    return (np_mean+V*(ub-lb)/n)[0]

### Bun 2019 trimmed mean

# assumes sorted data
def smooth_sens(data,m,t,a,b):
    n=len(data)
    cur_max=-1
    cur_max_2=-1
    for k in range(n+1):
        new=np.exp(-k*t)*cur_max_2
        if new>cur_max:
            cur_max=new
        cur_max_2=-1
        for ell in range(k+2):
            u_ind=n-m+1+k-ell
            if u_ind>n:
                U=b
            else:
                U=data[u_ind-1]
            l_ind=m+1-ell
            if l_ind<=0:
                L=a
            else:
                L=data[l_ind-1]
            new=(U-L)
            if new>cur_max_2:
                cur_max_2=new
    return cur_max/(n-2*m)




# binary search from Bun 2019 mean
def bs(f,a,b,tol=10**(-10)):
    cv=a
    curr=f(cv)
    if (curr<=0) and (f(b)>=0):
        while np.abs(curr)>tol:
            cv=(b+a)/2
            curr=f(cv)
            if curr<0:
                a=cv
            else:
                b=cv
        # print('sig ',cv)
        # print('f value ',f(cv))
    else:
        print('lb ',f(a))
        print('ub ',f(b))
        print("error")
    return cv
# cubic function from Bun 2019 mean
def cubic(sigma,c):
    return 5*(sigma**3)/c-5*(sigma**2)-1
# finding best parameters
def param_search(eps,t,printt):
    lb=t/eps
    ub=np.max((1/2,2*lb))
    sigma=bs(lambda x: cubic(x,lb),lb,ub)
    # sigma=bs(lambda x: lb*5*(x**3)-5*(x**2)-1,lb,ub)
    s=np.exp(-3*(sigma**2)/2)*(eps-t/sigma)
    return [sigma,s]

# param_search(1,10,printt=True)



# the new noise distribution from Bun 2019
def rlln(sigma):
    X=np.random.laplace(size=1)
    Y=np.random.normal(size=1)
    Z=X*np.exp(sigma*Y)
    return Z

def bun_2019(data, a,b, rho,m,t=0.1,seed=np.nan,printt=False):
    if seed is not np.nan:
        np.random.seed(seed=seed)
    n = len(data)   
    Z=np.clip(scipy.stats.trim_mean(data,m/n),a,b)
    sigma,s = param_search(np.sqrt(2*rho),t,printt)
    SS=smooth_sens(data,m,t,a,b)
    V=rlln(sigma)
    # print(Z)
    # print(SS)
    # print(SS)
    return (Z+V*SS/s)[0]


# noise is split up here
def bun_2019_split(data, a,b, rho,m,t=0.1,seed=np.nan,printt=False):
    if seed is not np.nan:
        np.random.seed(seed=seed)
    n = len(data)   
    Z=np.clip(scipy.stats.trim_mean(data,m/n),a,b)
    sigma,s = param_search(np.sqrt(2*rho),t,printt)
    SS=smooth_sens(data,m,t,a,b)
    V=rlln(sigma)
    # print(Z)
    # print(SS)
    # print(SS)
    return [Z,V*SS/s]

# concentrated dp private hist function
def CDPHist(data,rho,bins,range):
    c, bin_edges=np.histogram(data, bins=bins,range=range)
    # print(c)
    c=c+np.random.normal(size=len(c))/np.sqrt(2*rho/2)
    return [c, bin_edges]


# concentrated dp private hist helper
def CPDPRE(data,alpha,R,rho,k=2):
    # compute r and the number of bins - here alpha is error and k is the number of moments
    r=10/(alpha**(1/(k-1)))
    # gives -R-2r to R+2r by 2r increments
    bins= np.arange(-R - 2*r, R + 2*r + 1, 2*r)
    # print(bins)
    range=(-R - 2*r,R + 2*r+1)
    # Private DP histogram
    counts, bin_edges = CDPHist(data,rho,bins,range)
    index_of_max = np.argmax(counts)
    I=np.array([bin_edges[index_of_max]-2*r,2*r+bin_edges[index_of_max+1]])
    return I


# concentrated dp private hist mean
def CPDPMean(data,alpha,R,rho,beta=0.1):
    m=int(5*np.log(2/beta))
    n=len(data)
    j=int(n/2)
    k=int(j/m)
    # print(k)
    Z=data[0:(j+1)]
    W=data[(j+1):]
    means=np.zeros(m)
    for i in range(m):
        # print(int(i*k),int((i+1)*k-1))
        a=int(i*k)
        b=int((i+1)*k-1)
        Zi=Z[a:b]
        # Private DP histogram
        interval=CPDPRE(Zi,alpha,R,rho/2,k=2)
        ri=interval[1]-interval[0]
        Yi1=W[a:b]
        # print(Yi1)
        Yi=np.clip(Yi1,interval[0],interval[1])
        # print(np.mean(Yi))
        means[i]=np.mean(Yi)+np.sqrt(2/(2*rho))*np.random.normal(size=1)*ri/k
    return np.median(means)

# vanilla clipped mean
def clipped_mean(data,lb,ub,rho):
    n=len(data)
    Z=np.clip(data,lb,ub)
    np_mean=np.mean(Z)
    V=np.random.normal(scale=1/ np.sqrt(2*rho) , size=1)
    # print((ub-lb))
    return (np_mean+V*(ub-lb)/n)[0]

### Private winsorized mean
# pQ
def privateQuantile(data,q,eps,l,u):
    data=np.sort(data)
    Z=data-l
    n=len(data)
    llambda=u-l
    ZZ=np.zeros(n+2)
    ZZ[1:(n+1)]=Z
    ZZ[n+1]=llambda
    # print(ZZ[0])
    # print(ZZ[1])
    y=np.diff(ZZ)*np.exp(-eps*np.abs(np.arange(0,n+1)-q*n))
    p=y/np.sum(y)
    
    ind=np.random.choice(np.arange(0,n+1), size=1, replace=True, p=p)
    xi=np.random.uniform(ZZ[ind],ZZ[ind+1],1)+l
    return xi
    

# widened winsorized mean
def WWM(data,eps,l,u):
    Z=data-l
    n=len(data)
    rad=n**(1/3+1/10)
    # print(np.min(Z))
    a=privateQuantile(Z,1/4,eps/4,0,u-l)
    b=privateQuantile(Z,3/4,eps/4,0,u-l)
    mu1=(a+b)/2
    iqr1=np.abs(b-a)
    unew=mu1+4*rad*iqr1
    lnew=mu1-4*rad*iqr1
    clp=np.clip(Z,lnew,unew)
    V=np.random.laplace(scale=2/ eps , size=1)
    clp=np.mean(clp)+(unew-lnew)*V/n
    return clp+l
