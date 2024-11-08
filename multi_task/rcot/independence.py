"RCOT Code excerpt from Reimers"
import numpy as np

from scipy import stats
import numbers

from multi_task.rcot.momentchi2 import *


def gretton_heuristic(X, subset=500):
    """
    median || X_i - X_j ||_i!=j
    """
    if not subset is None:
        idx = np.random.choice(list(range(X.shape[0])), size=subset, replace=False)
        X = X[idx]

    norms = []

    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            norms.append(np.linalg.norm(X[i]-X[j]))

    sig = np.median(norms)
    if sig == 0:
        sig = 1
    return sig


"""
RCOT:
"""


def random_fourier_features(x,w=None,b=None,num_f=25,sigma=1):
    """
    https://github.com/ericstrobl/RCIT/blob/master/R/random_fourier_features.R
    """
    # check if x is a matrix/ has the right shape
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)

    r = x.shape[0]
    c = x.shape[1]

    if w is None:
        w = (1/sigma) * np.random.normal(size=(num_f, c))
        b = np.tile(2*np.pi*np.random.uniform(size=(num_f,1)), (1,r))

    feat = np.sqrt(2) * np.cos(w[:num_f,:c] @ x.T + b[:num_f,:]).T

    return feat, w, b


def cov(x, y=None):
    if y is None:
        return np.cov(x, rowvar=False)
    else:
        c = np.cov(x,y, rowvar=False)
        temp = c[:x.shape[1], -y.shape[1]:]
        if x.shape[1] == 1:
            return temp[0]
        else:
            return temp


def Sta_perm(f_x,f_y,r):
    Cxy = cov(f_x,f_y)
    Sta = r*np.sum(np.square(Cxy))
    return Sta




def rcot(X, Y, Z, approx="lpd4", num_f=100, num_f2=5, subset=1000):
    """
    Source: https://github.com/ericstrobl/RCIT/blob/master/R/RCoT.R
    """
    assert X.shape[0] == Y.shape[0] == Z.shape[0]

    if not subset is None:
        idx = np.random.choice(list(range(X.shape[0])), size=subset, replace=False)
        X = X[idx]
        Y = Y[idx]
        Z = Z[idx]

    # that is the matrix conversion part
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=1)
        Y = np.expand_dims(Y, axis=1)
        if len(Z.shape) == 1:
            Z = np.expand_dims(Z, axis=1)

    # now let us exclude all constant columns in Z
    Z = Z[:, np.std(Z,axis=0)>0]
    d = Z.shape[1]

    # if Z is completely constant run hsic
    if Z.shape[1] == 0:
        print("Z is constant! Unconditional independence test is needed!")
        exit()
    # if either X or Y are constant return p value 1
    elif np.std(X) == 0 or np.std(Y) == 0:
        return float("NaN"),1

    r = X.shape[0]
    r1 = 500 if r > 500 else r

    # normalize to zero mean and unit variance in each column
    X = (X - X.mean(axis=0))/X.std(axis=0)
    Y = (Y - Y.mean(axis=0))/Y.std(axis=0)
    Z = (Z - Z.mean(axis=0))/Z.std(axis=0)

    # the sigma is again the gretton heuristic
    four_Z = random_fourier_features(Z[:,:d],num_f=num_f,sigma=gretton_heuristic(Z, subset=r1))
    four_X = random_fourier_features(X,num_f=num_f2,sigma=gretton_heuristic(X, subset=r1))
    four_Y = random_fourier_features(Y,num_f=num_f2,sigma=gretton_heuristic(Y, subset=r1))

    f_X =  (four_X[0] - four_X[0].mean(axis=0))/four_X[0].std(axis=0)
    f_Y =  (four_Y[0] - four_Y[0].mean(axis=0))/four_Y[0].std(axis=0)
    f_Z =  (four_Z[0] - four_Z[0].mean(axis=0))/four_Z[0].std(axis=0)

    # numpy.cov does not exactly reproduce R cov()
    # first we need to set the rowvar flag to False
    # then it is equivalent for a single matrix
    # for two matrizes we need to extract the upper right quadrant from the numpy.cov 
    #>>> x = np.array([0.1, 3, 2, -5, -1, 0]).reshape(2,3).T
    #>>> y= np.array([4,-2,3.4,9,0,-1]).reshape(2,3).T
    #>>> x
    #array([[ 0.1, -5. ],
    #    [ 3. , -1. ],
    #    [ 2. ,  0. ]])
    #>>> y
    #array([[ 4. ,  9. ],
    #    [-2. ,  0. ],
    #    [ 3.4, -1. ]])
    #>>> np.cov(x,y, rowvar=False)
    #array([[  2.17      ,   3.35      ,  -3.99      ,  -7.35      ],
    #    [  3.35      ,   7.        ,  -3.6       , -14.5       ],
    #    [ -3.99      ,  -3.6       ,  10.92      ,   9.1       ],
    #    [ -7.35      , -14.5       ,   9.1       ,  30.33333333]])
    #
    # while R returns:
    # [[-3.99, -7.35],
    #  [-3.6, -14.5]]
    #
    # to be even more specific we need to catch the right block in cov for matrizes of different sizes
    #>>> y= np.array([4,-2,3.4,9,0,-1,0.33,5,5]).reshape(3,3).T
    #>>> x = np.array([0.1, 3, 2, -5, -1, 0]).reshape(2,3).T
    #>>> x
    #array([[ 0.1, -5. ],
    #    [ 3. , -1. ],
    #    [ 2. ,  0. ]])
    #>>> y
    #array([[ 4.  ,  9.  ,  0.33],
    #    [-2.  ,  0.  ,  5.  ],
    #    [ 3.4 , -1.  ,  5.  ]]
    #>>> cov(x,y)
    #numpy
    #[[  2.17         3.35        -3.99        -7.35         3.736     ]
    #[  3.35         7.          -3.6        -14.5          7.005     ]
    #[ -3.99        -3.6         10.92         9.1         -5.137     ]
    #[ -7.35       -14.5          9.1         30.33333333 -14.78833333]
    #[  3.736        7.005       -5.137      -14.78833333   7.26963333]]
    # R cov
    #array([[ -3.99 ,  -7.35 ,   3.736],
    #       [ -3.6  , -14.5  ,   7.005]])

    Cxy = cov(f_X,f_Y)
    Czz = cov(f_Z)

    # now calculate the inverse (cholesky decomposition because it is faster)
    temp = np.linalg.inv(np.linalg.cholesky(Czz + 1e-10*np.identity(num_f)))
    i_Czz = np.dot(np.transpose(temp),temp)
    #i_Czz = np.linalg.inv(Czz)

    Cxz = cov(f_X,f_Z)
    Czy = cov(f_Z,f_Y)

    z_i_Czz = f_Z @ i_Czz
    e_x_z = z_i_Czz @ Cxz.T
    e_y_z = z_i_Czz @ Czy

    #approximate null distributions
    res_x = f_X-e_x_z
    res_y = f_Y-e_y_z

    if num_f2==1:
        approx="hbe"

    if approx == "perm":
        Cxy_z = cov(res_x, res_y)
        Sta = r*np.sum(np.square(Cxy_z))

        nperm =1000
        Stas = []
        for ps in range(nperm):
            perm = np.array(range(r))
            np.random.shuffle(perm)
            Sta_p = Sta_perm(res_x[perm,:],res_y,r)
            Stas.append(Sta_p)
        
        p = 1-(np.sum(Sta >= Stas)/len(Stas))

    else:
        Cxy_z=Cxy-Cxz @ i_Czz @ Czy #less accurate for permutation testing
        Sta = r*np.sum(np.square(Cxy_z))

        d = np.array(np.meshgrid(range(f_X.shape[1]), range(f_Y.shape[1]))).reshape(2, f_X.shape[1]*f_Y.shape[1]).T

        res = res_x[:,d[:,0]]*res_y[:,d[:,1]]
        Cov = 1/r * (res.T @ res)

        if approx == "chi2":
            i_Cov = np.linalg.inv(Cov)

            flat_Cxy_z = Cxy_z.T.reshape((1, Cxy_z.shape[0]*Cxy_z.shape[1]))
            Sta = r * (flat_Cxy_z @  i_Cov @ flat_Cxy_z.T )
            Sta = Sta[0][0]

            p = 1-stats.chi2.cdf(Sta, Cxy_z.shape[0]*Cxy_z.shape[1])
        else:
            e_val, e_vec = np.linalg.eig(Cov)

            if approx == "gamma":
                p = 1-sw(e_val, Sta)

            elif approx == "hbe":
                p = 1-hbe(e_val, Sta)

            elif approx == "lpd4":
                try:
                    p = 1-lpb4(e_val, Sta)
                except:
                    print("lpb4 failed using hbe now!")
                    p = 1-hbe(e_val, Sta)

                if isinstance(p, numbers.Number) or p is np.NaN:
                    print("lpb4 returned NaN, running hbe now!")
                    p=1-hbe(e_val, Sta)
            else:
                print("Use one of the following approximation methods: ")
                print("perm, chi2, gamma, hbe, lpd4")
                exit()
    if p < 0:
        p = 0
    return Sta, p, np.linalg.det(Cxy_z)

def rcot_100(x, y, c, subset=None, approx="lpd4"):
    return rcot(x, y, c, num_f = 100, num_f2 = 100, subset=subset, approx=approx)

