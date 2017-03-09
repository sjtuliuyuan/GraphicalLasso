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
beta_history = EM.beta_history
loglikelihood_history = EM.loglikelihood_history
print(loglikelihood_history)
MSE=[]
for i in beta_history:
	MSE.append(0.5*np.linalg.norm(i.reshape(2,1)-beta.reshape(2,1))**2)



fig = plt.figure()

ax1 = fig.add_subplot(111)
line1, =ax1.plot(MSE,'b', label="MSE")
ax1.set_ylabel('Mean Squre Error')
ax1.set_title("MSE and log-likelihood")

ax2 = ax1.twinx()  # this is the important function
line2, =ax2.plot(loglikelihood_history, 'r', label = "log-likelihood")
ax2.set_ylabel('log-likelihood')
ax2.set_xlabel('Number of Iteration Steps')
ax2.legend(["MSE: 'blue' ",
            "log-likelihood:, 'red'"],
           loc='upper left')



first_legend = plt.legend(handles=[line1,line2], loc=1)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

# Create another legend for the second line.

plt.show()