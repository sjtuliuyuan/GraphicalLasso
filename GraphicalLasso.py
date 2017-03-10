import numpy as np
import sklearn.covariance as sk
from sklearn.linear_model import cd_fast
from sklearn.utils.extmath import pinvh
from sklearn.covariance import empirical_covariance
from sklearn.utils.validation import check_random_state
import matplotlib.pyplot as plt



class GraphLasso():
	def __init__(self,X=None,tol=1e-4, max_iter=100):
		self.X=X   
		self.tol = tol
		self.max_iter=int(max_iter)

	def fit(self, X, alpha):
		self.alpha=alpha
		emp_cov = empirical_covariance(X)
		self.covariance_, self.precision_ = graph_lasso(
            emp_cov, alpha=self.alpha, tol=self.tol, max_iter=self.max_iter)
		return self.covariance_, self.precision_

def graph_lasso(emp_cov, alpha, tol=1e-4, max_iter=100):

    _, n_features = emp_cov.shape    
    covariance_ = emp_cov.copy()

    covariance_ *= 0.95
    diagonal = emp_cov.flat[::n_features + 1]
    covariance_.flat[::n_features + 1] = diagonal
    precision_ = pinvh(covariance_)

    indices = np.arange(n_features)
    eps=np.finfo(np.float64).eps

    for i in range(max_iter):
        for idx in range(n_features):
            sub_covariance = np.ascontiguousarray(
                    covariance_[indices != idx].T[indices != idx])
            row = emp_cov[idx, indices != idx]

            # Use coordinate descent
            coefs = -(precision_[indices != idx, idx]
                          / (precision_[idx, idx] + 1000 * eps))
            coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                            coefs, alpha, 0, sub_covariance, row, row,
                            max_iter, tol, check_random_state(None), False)

            # Update the precision matrix
            precision_[idx, idx] = (
                    1. / (covariance_[idx, idx]
                          - np.dot(covariance_[indices != idx, idx], coefs)))
            precision_[indices != idx, idx] = (- precision_[idx, idx]
                                                   * coefs)
            precision_[idx, indices != idx] = (- precision_[idx, idx]
                                                   * coefs)
            coefs = np.dot(sub_covariance, coefs)
            covariance_[idx, indices != idx] = coefs
            covariance_[indices != idx, idx] = coefs
    return covariance_, precision_


def get_precision_recall():
	golden_percision = np.array([[1,1,1,0],
							[1,1,1,1],
							[1,1,1,1],
							[0,1,1,1]])

	X = np.loadtxt('graph.csv',delimiter=',',dtype='float')
	gl = GraphLasso(tol=1e-10,max_iter=100)

	alpha_set=[1e-5,1e-4,1e-4,1e-2,1e-1]
	recall_list = []
	precision_list = []
	for alpha in alpha_set:
		covariance_, precision_ = gl.fit(X,alpha= alpha)
		print("Precision Matrix")
		print(precision_)

		TPFP = len(np.nonzero(precision_)[0])
		TP = len(np.nonzero(np.multiply(precision_, golden_percision))[0])
		pre = float(TP)/TPFP
		TPFN = len(np.nonzero(golden_percision)[0])
		recall = float(TP)/TPFN
		print("Precision")
		print(pre)
		print("Recall")
		print(recall)
		precision_list.append(pre)
		recall_list.append(recall)
	return precision_list, recall_list

def plot_picture(precision_list, recall_list):
	line1, = plt.semilogx([1e-5,1e-4,1e-4,1e-2,1e-1],precision_list, 'r',label='precision')
	line2, = plt.semilogx([1e-5,1e-4,1e-4,1e-2,1e-1],recall_list,'b',label='recall')
	plt.axis([0, 0.1, 0, 1.2])
	first_legend = plt.legend(handles=[line1,line2], loc=1)
	ax = plt.gca().add_artist(first_legend)
	plt.xlabel('lambda')
	plt.title("Precision and Recall")
	plt.show()



precision_list, recall_list = get_precision_recall()
plot_picture(precision_list, recall_list)

