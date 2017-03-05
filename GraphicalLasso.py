"""
============================================================
Graphical Lasso
This algorithm is based on the paper :
		Sparse inverse covariance estimation with the graphical lasso
The get and put function in class GraphLasso is the tricky part.
============================================================

"""
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np


class GraphLasso:
    # X is data (n_samples*n_features)
    # A is precision matrix (n_features*n_features)
    # S is covariance matrix (n_features*n_features)
    # rho is regularizer
    
    # initialization
    def __init__(self,X=None,A=None,S=None,rho=0.1,
                 maxItr=100):
        self.X=X   
        self.rho=rho
        self.maxItr=int(maxItr)
        
    # graphical lasso    
    def fit(self,X):
        n_samples,n_features=X.shape[0],X.shape[1]
        
        # remove the mean and scale to unit varince
        scaler=preprocessing.StandardScaler().fit(X)
        self.X=scaler.transform(X)
        
        #initialize S(empirical covariance) and W(estimated covariance)
        S=self.X.T.dot(self.X)/n_samples
        self.S=S
        W=S+self.rho*np.eye(self.X.shape[1])
        #initialize A, A is the precision matrix
        #A = np.eye(X.shape[1])
        A=np.linalg.pinv(S)

        clf=linear_model.Lasso(alpha=self.rho)
        for i in range(self.maxItr):
            for j in range(n_features):
            	#The get and put function in class GraphLasso is the tricky part.
                W11,w12,w22=self.get(W)
                S11,s12,s22=self.get(S)
                A11,a12,a22=self.get(A)

                #calculate W11^{-1/2}            
                U,D,V=np.linalg.svd(W11)
                W11_half=U.dot(np.diag(np.sqrt(D)).dot(U.T))
                
                b=np.linalg.pinv(W11_half).dot(s12)
                
                # performs lasso  
                beta=clf.fit(W11_half,b).coef_
            
                # find w12
                w12=W11.dot(beta)
                
                #formula (13) (14) of the paper
                a12=-beta/(w22-beta.T.dot(W11).dot(beta))
                a22=1/(w22-beta.T.dot(W11).dot(beta))
            
                W=self.put(W11,w12,w22)
                S=self.put(S11,s12,s22)
                A=self.put(A11,a12,a22)
        self.A=A

        return self
             
    def get(self,S):
    	# S = | R    s  |
    	#	  | s^T  sii|
        end=S.shape[0]-1
        R=S[:-1,:-1]
        s=S[end,:-1]
        sii=S[end][end]
            
        return [R,s,sii]
    
    def put(self,R,s,sii):
    	#return X =	| sii  s^T |
    	#         	| s    R   |	
        n=R.shape[0]+1
        X=np.empty([n,n])
        X[1:,1:]=R
        X[1:,0]=s
        X[0,1:]=s
        X[0][0]=sii
        
        return X