"""
============================================================
EMalgorithm:
This algorithm is an EM Algorithm for Confounded Heterogenous Data
		y \sim X\beta +Zu+\epsilon
		u \sim N(0, I \sigma_u^2)
		Y \sim N(X\beta, ZZ^T \epsilon_\sigma^2 + I \sigma_u^2)
============================================================

"""
import numpy as np
import math

class EM:
    # y X Z is the observed data
    # maxItr is the maximum iteration steps
    # initialization
    def __init__(self,maxItr=100):
        self.maxItr=int(maxItr)

    def fit(self,y=None,X=None,Z=None):
    	self.y = y   
        self.X = X
        self.Z = Z
        self.K = Z.dot(Z.T)
        self.n,self.p = X.shape
        self.q = Z.shape[1]
        self.y = self.y.reshape(self.n,1)
        self.beta = np.zeros((self.p, 1))
        self.sigma_u = 0.1
        self.sigma_epsilon = 0.1
        self.beta_history=[]
        self.loglikelihood_history=[]
        for _ in xrange(0, self.maxItr):
        	u,utu = self.Estep()
        	self.Mstep(u,utu)
        	self.beta_history.append(self.beta)
        	self.loglikelihood_history.append(self.get_loglikelihood(u,utu))

    def Estep(self):
    	#print("_____EStep______")
    	Omega_inverse = np.linalg.pinv(self.sigma_u**2*self.K
    						+self.sigma_epsilon**2*np.eye(self.n) )
    	u = self.sigma_u**2*self.Z.T.dot(Omega_inverse).dot(
    						 self.y-self.X.dot(self.beta))
    	Tr = self.sigma_u**2*np.eye(self.q)-self.sigma_u**4\
    						*self.Z.T.dot(Omega_inverse).dot(self.Z)
    	Tr = np.trace(Tr)
    	norm = self.sigma_u**2*self.Z.T.dot(Omega_inverse).dot(
    							self.y-self.X.dot(self.beta))
    	norm = np.linalg.norm(norm)**2
    	utu = Tr+norm
    	return u,utu	
    
    def Mstep(self, u, utu):
    	#print("_____MStep______")
    	self.sigma_u = math.sqrt(utu/self.q)
    	temp = self.y - self.X.dot(self.beta) - self.Z.dot(u)
    	self.sigma_epsilon = math.sqrt((temp.T.dot(temp))/self.n)
    	self.beta = np.linalg.pinv(self.X.T.dot(
    			self.X)).dot(self.X.T).dot(self.y-self.Z.dot(u))

    def get_loglikelihood(self, u, utu):
    	temp = self.y-self.X.dot(self.beta)-self.Z.dot(u) 
    	LE = -self.n*math.log(self.sigma_epsilon)-self.q*math.log(self.sigma_u)\
    			-temp.T.dot(temp)/(2*self.sigma_u**2)\
    			-utu/(2*self.sigma_u**2)
    	return float(LE)



