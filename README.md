# This is the PGM homework2 problem 2.

``GraphicalLass.py'' is based on the paper:
		Sparse inverse covariance estimation with the graphical lasso
You can run it by the following code(A is the percision matrix):
	import numpy as np
	import GraphicalLasso as GL
	X = np.loadtxt('graph.csv',delimiter=',',dtype='float')
	rho = 1e-5
	gl=GL.GraphLasso(rho)
	gl.fit(X)
	print(gl.A)
Or you can run it by python GraphicalLasso_run.py




``Kronecker_product_one.py'' contains the function used in problem 2.2.3.
	the input of function get_GL_condition(c) is c, it well return the value of
			max||(Omega \kr \Omega)_{eS} (Omega \kr \Omega)^{-1}_{S,S}||_1
You can run it by python Kronecker_product_one.py



``Kronecker_product_two.py'' contains the function used in problem 2.3.2.
	the input of function get_GL_condition(c) is c, it well return the value of
			max||Tau_{e,S} Tau_{S,S}^{-1}||_1
You can run it by python Kronecker_product_two.py



