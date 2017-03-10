import numpy as np
import EMalgorithm
import matplotlib.pyplot as plt

data = np.loadtxt('ToothGrowth.csv', delimiter=',', dtype='float')
y = data[:,1].reshape(60,1)
X = data[:,3].reshape(60,1)
Z = data[:,2].reshape(60,1)

print('The dimension of y is:'+str(y.shape))
print('The dimension of X is:'+str(X.shape))
print('The dimension of Z is:'+str(Z.shape))

EM = EMalgorithm.EM(maxItr=100)
EM.fit(y=y,X=X,Z=Z)
beta_history = EM.beta_history
sigma_history = EM.sigma_epsilon_history
beta=[]
sigma=[]
for b in beta_history:
	beta.append(float(b))
for s in sigma_history:
	sigma.append(float(s))

linear_beta=float(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y))
linear_sigma = np.var(y-beta*X)

print("linear model beta:"+str(linear_beta))
print("linear model sigma:"+str(linear_sigma))
print(sigma)



#line1, =plt.plot(beta,'b', label="confounding model beta")
#line2, =plt.plot([x for x in range(0,100)],[linear_beta for x in range(0,100)],'r', label='vanilla linear model beta')
#plt.legend(handles=[line1,line2], loc=1)
#plt.xlabel("number of Iteration")
#plt.title("Figure 1, beta")
#plt.show()
line1, =plt.plot(sigma,'b', label="confounding model sigma")
line2, =plt.plot([x for x in range(0,100)],[linear_sigma for x in range(0,100)],'r', label='vanilla linear model sigma')
plt.legend(handles=[line1,line2], loc=1)
plt.xlabel("number of Iteration")
plt.title("Figure 2, sigma")
plt.show()