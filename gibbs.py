# coding: utf-8

# In[488]:

#package imports
import numpy as np
import scipy as sp
import scipy.stats as stats

class beta:
    #follows multivariate normal
    def __init__(self, y,x,beta_0, Sigma_0,p, sigma_sq):
        n = y.shape[0]
        #construct c_p
        C_p = np.empty((n,n))
        #fill it with the powers of p
        for i in range(0,n):
            for j in range(0,n):
                C_p[i,j] = p**(max(i-j,j-i))
        tau = np.linalg.inv(sigma_sq*C_p)
        #XT τX + Σ^{−1}
        V_inverse = np.dot(x.T,np.dot(tau,x)) + np.linalg.inv(Sigma_0)
        V = np.linalg.inv(V_inverse)
        #m = V (X.T τy + Σ^{−1}μ).
        
        mean = np.dot(V,np.dot(x.T,np.dot(tau,y)) + np.dot(np.linalg.inv(Sigma_0),beta_0))
        #these are subject to change given new mu, sig, etc
        self.mean =mean
        self.cov = V
        #not going to change on update - so might as well save
        self.y = y
        self.x = x
        self.beta_0 = beta_0
        self.Sigma_0 = Sigma_0
        
    
    def sample(self):
        samp = np.random.multivariate_normal(self.mean, self.cov)
        return samp
    
    def update(self, sigma_sq, p):
        #resets attributes for new prior params
        self.__init__(self.y, self.x, self.beta_0, self.Sigma_0,p,sigma_sq)
        return self
 
class rho:

    def __init__(self, epsilon, x,y, center, beta_0, Sigma_0, v_0, sigma_0_sq):
        #save all these constants
        self.epsilon = epsilon
        self.x = x
        self.y = y
        self.center = center

    def construct_pmat(self, p):
        n = self.y.shape[0]
        #construct c_p
        C_p = np.empty((n,n))
        #fill it with the powers of p
        for i in range(0,n):
            for j in range(0,n):
                C_p[i,j] = p**(max(i-j,j-i))
        return C_p

    def sample(self, beta, sigma_sq):
        #draw value J(\theta_t | \theta_t-1)
        if self.center+ self.epsilon > 1:
            cand = np.random.uniform(1-2*self.epsilon,1)
        elif self.center- self.epsilon < 0:
            cand = np.random.uniform(0,2*self.epsilon)
        else:
            cand = np.random.uniform(self.center-self.epsilon,self.center+self.epsilon)

        #compute unnormalized probability
        cp = self.construct_pmat(cand)
        r_top = stats.multivariate_normal.pdf(self.y,mean=np.dot(self.x,beta),cov=sigma_sq*cp)
        cp_last = self.construct_pmat(self.center)
        r_bottom = stats.multivariate_normal.pdf(self.y,mean=np.dot(self.x,beta),cov=sigma_sq*cp_last)
        r = float(r_top)/r_bottom
        u = np.random.uniform(0,1)
        stats.multivariate_normal.pdf(self.y,mean=np.dot(self.x,beta),cov=sigma_sq*cp*.001)
        #print "comp mean", np.dot(self.x,beta)
        #print "comp cov", sigma_sq*cp
        if u < r:
            self.center = cand
        return self.center


class sig_sq:
    #follows an inverse gamma dstribution
    def __init__(self,v_0, sigma_0_sq, y, x, beta, p):
        #construct p mat
        self.y = y
        n = self.y.shape[0]
        #construct c_p
        C_p = np.empty((n,n))
        #fill it with the powers of p
        for i in range(0,n):
            for j in range(0,n):
                C_p[i,j] = p**(max(i-j,j-i))
        shape = .5*(v_0 + y.shape[0])
        y_xb = y - np.dot(x,beta)
        scale = .5*((4./v_0*sigma_0_sq) + np.dot(y_xb.T,np.dot(np.linalg.inv(C_p), y_xb)))
        self.shape =shape
        self.scale = scale
        self.v0 = v_0
        self.sig_0 = sigma_0_sq
        self.x =x
        
    def sample(self):
        return 1./np.random.gamma(shape=self.shape, scale = 1./self.scale)
    
    
    #follows an inverse gamma dstribution
    def update(self, beta, p):
        self.__init__(self.v0, self.sig_0, self.y, self.x, beta,p)

        
class hyperparameters:
    def __init__(self, y, x):
        
        ols_beta = np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y))
        resid_mean = np.mean(np.square(y - np.dot(x, ols_beta)))
        #saved all these hyperparams
        self.sigma_sq_0 = np.var(x)/3.
        self.beta_0 = np.zeros(x.shape[1])
        self.v_0=3
        self.Sigma_0 = np.diag(100*ols_beta)
        

class Gibbs:       
    def sample(self,y, x,n_samples):
        #Fix the initial hyperparameters
        hypers= hyperparameters(y,x)
        #initialize our objects
        #some of these values will be overwritten
        beta_ob = beta(y,x,hypers.beta_0, hypers.Sigma_0,.5,hypers.sigma_sq_0)
        last_beta = beta_ob.sample()
        sig_ob = sig_sq(hypers.v_0, hypers.sigma_sq_0, y, x, last_beta, .5)
        last_sig = sig_ob.sample()
        rho_ob =rho(.15, x,y,.2,np.zeros(x.shape[1]),hypers.Sigma_0,hypers.v_0,hypers.sigma_sq_0)
        last_rho= rho_ob.sample(last_beta, last_sig)

        returns = [[np.copy(last_beta), np.copy(last_sig), np.copy(last_rho)]]
        
        for samp in range(n_samples):
            #1. sample beta
            beta_ob.update(last_sig, last_rho)
            last_beta = np.copy(beta_ob.sample())
            #2.  Sample little sigma
            sig_ob.update(last_beta,last_rho)
            last_sig = sig_ob.sample()
            #3.  Sample rho
            last_rho = rho_ob.sample(last_beta, last_sig)
            #add samples to our return
            returns.append([np.copy(last_beta), np.copy(last_sig), np.copy(last_rho)])
            #print "iter: ", samp
            #print returns[-1]
        return returns

