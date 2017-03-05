# -*- coding: utf-8 -*-

"""
pgm homework2 problem 2
"""
import numpy as np
import GraphicalLasso as GL

#load the data_set
X = np.loadtxt('graph.csv',delimiter=',',dtype='float')


rho = 1e-5
gl=GL.GraphLasso(rho)
gl.fit(X)

print(gl.A)


