import numpy as np 
import matplotlib.pyplot as plt

def get_GL_condition(c=0.1):

	Omega =  np.array([	[1,     c,    0,  2*c*c],
						[c,     1,    c,  c	   ],
						[0,     c,    1,  c    ],
						[2*c*c, c,    c,  1    ]])

	# Kronecker product of Omega \kr Omega
	KP=np.kron(Omega,Omega)
	# calculate S, it will be a list
	edge_set = [[1,2],[2,1],[1,3],[3,1],[3,4],[4,3],[2,4],[4,2],[2,3],[3,2]]
	Slist=[]
	for [a,b] in edge_set:
		Slist.append((a-1)*Omega.shape[0]+(b-1))
	Slist.sort()
	
	# calculate (Omega \kr Omega)_{S,S}^{-1}
	KP_SS_inverse=np.linalg.inv(KP[Slist,:][:,Slist])

	SC = {3,12}
	val =[]
	for e in SC:
		KP_eS=KP[e,:][Slist]
		val.append(np.sum(np.abs(KP_eS.dot(KP_SS_inverse))))
	return max(val)


C=[x/100.0 for x in range(1,70)]
Val=[]
for c in C:
	Val.append(get_GL_condition(c)
)
max_c_position = max([Val.index(x) for x in Val if x<1])
max_C = (max_c_position+1)*0.01
print(max_C)

plt.plot(C, Val, 'ro')
plt.plot([0.01*x for x in range(0,max_c_position+2)], [1 for x in range(0,max_c_position+2)],'b')
plt.plot([max_C for x in range(0,11)], [0.1*x for x in range(0,11)],'b')
plt.axis([0, 0.5, 0, 2])
plt.xlabel('c')
plt.ylabel('max||(Omega \kr \Omega)_{eS} (Omega \kr \Omega)^{-1}_{S,S}||_1')
plt.show()

