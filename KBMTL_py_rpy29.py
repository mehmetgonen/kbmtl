"""
Python wrapper based on KBMTL in R
For rpy2 earlier versions (tested on 2.9)

Requires rpy2, sklearn, pandas

Default parameters:
    KBMTL:
        par = {'alpha_lambda':1,'beta_lambda':1,'alpha_epsilon':1,'beta_epsilon':1,
                'iteration':20,'R':20,'sigma_w':1.0,'sigma_h':0.1}
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4147917/
            # we have considered 200 iterations and gamma prior values (both α and β) of one. Subspace dimensionality has been considered to be 20,
            # and the standard deviation of hidden representations and weight parameters are selected to be the defaults 0.1 and one, respectively.
            # https://www.mdpi.com/1999-4893/9/4/77/htm
    kernel type:
        RBF, scale = 1
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html

If rpy2 cannot be imported due to OSError: cannot load library...
    add R to path manually
    os.environ["R_HOME"] = '.../Anaconda3/envs/{env name}/lib/R/'
    os.environ["PATH"] = ".../Anaconda3/envs/{env name}/lib/R/bin/x64/" + ";" + os.environ["PATH"]

Example:
    from KBMTL_py_rpy29 import KBMTL_R
    mdl = KBMTL_R()
    mdl.fit(x, y)
    prediction = mdl.predict(x_test)
"""


import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import linear_kernel


import rpy2.robjects as ro  # pip install rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import globalenv

class KBMTL_R():
    def __init__(self, kernel_type='rbf', kernel_scale=1, kbmtl_params=None):
        """
        Params:
        kernel_type: the kernel used to compute kernel matrix K of X.
        kernel scale:
        the standard deviation of Gaussian function. See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html
        kbmtl_params:
        the parameters of KBMTL. Must be in dictionary format.
        """
        if kbmtl_params is None:
            par ={'alpha_lambda':1,'beta_lambda':1,'alpha_epsilon':1,'beta_epsilon':1,
                    'iteration':20,'R':20,'sigma_w':1.0,'sigma_h':0.1}
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4147917/
            # we have considered 200 iterations and gamma prior values (both α and β) of one. Subspace dimensionality has been considered to be 20,
            # and the standard deviation of hidden representations and weight parameters are selected to be the defaults 0.1 and one, respectively.
            # https://www.mdpi.com/1999-4893/9/4/77/htm

        self.par = pd.DataFrame(data=par,index=[0])

        # Self.kernel should be a callable.
        if kernel_type == 'rbf':
            self.kernel = RBF(kernel_scale)
        elif kernel_type == 'linear':
            self.kernel = linear_kernel
        else:
            raise ValueError("Other kernels to be done. ")

        return None

    def fit(self, xx, yy):
        r = ro.r
        r.source('kbmtl_semisupervised_regression_variational_train.R')
        kbmtl_semisupervised_regression_variational_train = r.kbmtl_semisupervised_regression_variational_train
        pandas2ri.activate()
        # Ktrain should be an Ntra x Ntra matrix containing similarity values between training samples
        Ktrain = self.kernel(xx)
        Ytrain = np.asarray(yy)
        print("KBMTL model fitting...")
        self.state = kbmtl_semisupervised_regression_variational_train(Ktrain, Ytrain, self.par)
        self._isfitted = True
        self.x_train = xx
        return self

    def predict(self, x_test):
        assert (self._isfitted)
        r1 = ro.r
        r1.source('kbmtl_semisupervised_regression_variational_test.R')
        kbmtl_semisupervised_regression_variational_test =  r1.kbmtl_semisupervised_regression_variational_test
        pandas2ri.activate()
        Ktest = self.kernel(self.x_train, x_test)
        pred = kbmtl_semisupervised_regression_variational_test(Ktest,self.state)
        # d  = { key : pred.rx2(key)[0] for key in pred.names }
        # Y = d['Y']
        # Y = pandas2ri.ri2py(Y)
        Y = pandas2ri.ri2py(pred)
        return Y


if __name__ == "__main__":
    # Test code.
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor as RF

    x_train = np.random.randn(20, 5)
    x_test = np.random.randn(10, 5)
    A = np.random.randn(5, 10)
    y_train = x_train @ A
    y_train = y_train + 0.05 * np.random.randn(*y_train.shape)
    y_test = x_test @ A

    mdl = KBMTL_R('rbf', 0.5)
    mdl.fit(x_train, y_train)
    pred = mdl.predict(x_test)
    print(mean_squared_error(y_test, pred))

    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)

    mdl = KBMTL_R('linear')
    mdl.fit(x_train, y_train)
    pred = mdl.predict(x_test)
    print(mean_squared_error(y_test, pred))

    print("Reference RF:")
    mdl = RF()
    mdl.fit(x_train, y_train)
    pred = mdl.predict(x_test)
    print(mean_squared_error(y_test, pred))
