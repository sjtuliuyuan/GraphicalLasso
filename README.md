# This is the PGM homework2
``Emalgorithm.py''
This algorithm is an EM Algorithm for Confounded Heterogenous Data
		y \sim X\beta +Zu+\epsilon
		u \sim N(0, I \sigma_u^2)
		Y \sim N(X\beta, ZZ^T \epsilon_\sigma^2 + I \sigma_u^2)
You can test it by python Emalgorithm_run.py



``GraphicalLass.py'' is based on the paper:
		Sparse inverse covariance estimation with the graphical lasso
You can run it by python GraphicalLasso_run.py




``Kronecker_product_one.py'' contains the function used in problem 2.2.3.
	the input of function get_GL_condition(c) is c, it well return the value of
			max||(Omega \kr \Omega)_{eS} (Omega \kr \Omega)^{-1}_{S,S}||_1
You can run it by python Kronecker_product_one.py



``Kronecker_product_two.py'' contains the function used in problem 2.3.2.
	the input of function get_GL_condition(c) is c, it well return the value of
			max||Tau_{e,S} Tau_{S,S}^{-1}||_1
You can run it by python Kronecker_product_two.py



