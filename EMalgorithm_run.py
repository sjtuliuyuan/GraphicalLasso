import numpy as np
import EMalgorithm
import matplotlib.pyplot as plt


y = np.loadtxt('y.csv',delimiter=',',dtype='float')
X = np.loadtxt('X.csv',delimiter=',',dtype='float')
Z = np.loadtxt('Z.csv',delimiter=',',dtype='float')
beta = np.loadtxt('beta.csv',delimiter=',',dtype='float')

print('The dimension of y is:'+str(y.shape))
print('The dimension of X is:'+str(X.shape))
print('The dimension of Z is:'+str(Z.shape))

EM = EMalgorithm.EM(maxItr=100)
EM.fit(y=y,X=X,Z=Z)
history = EM.history
MSE=[]
for i in history:
	MSE.append(0.5*np.linalg.norm(i.reshape(2,1)-beta.reshape(2,1))**2)

plt.plot(MSE)
plt.xlabel('Number of Interation Steps')
plt.ylabel('Mean Square Error')
plt.show()