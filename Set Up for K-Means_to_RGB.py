import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import time

def plotCurrent(X, Rnk, Kmus):
    N, D = X.shape
    K = Kmus.shape[0]

    InitColorMat = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [0, 0, 0],
                             [1, 1, 0],
                             [1, 0, 1],
                             [0, 1, 1]])

    KColorMat = InitColorMat[0:K,:]

    colorVec = np.dot(Rnk, KColorMat)
    muColorVec = np.dot(np.eye(K), KColorMat)
    plt.scatter(X[:,0], X[:,1], c=colorVec)

    plt.scatter(Kmus[:,0], Kmus[:,1], s=200, c=muColorVec, marker='d')
    plt.axis('equal')
    plt.show()


def runKMeans(K,fileString):
    #load data file specified by fileString from Bishop book
    X = np.loadtxt(fileString, dtype='float')

    #determine and store data set information
    N, D = X.shape

    #allocate space for the K mu vectors
    Kmus = np.zeros((K, D))

    #initialize cluster centers by randomly picking points from the data
    rand_inds = np.random.permutation(N)
    Kmus = X[rand_inds[0:K],:]

    #specify the maximum number of iterations to allow
    maxiters = 1000

    for iter in range(maxiters):
        #assign each data vector to closest mu vector as per Bishop (9.2)
        #do this by first calculating a squared distance matrix where the n,k entry
        #contains the squared distance from the nth data vector to the kth mu vector
        def calcSqDistances(X, KMu):
            N,K = np.shape(X)
            sqDist = np.zeros((N,K), dtype = np.float32)
            for i in range(N):
                for j in range(K):
                    sqDist[i,j] = np.linalg.norm(X[i]- KMu[j])
            return sqDist
        #sqDmat will be an N-by-K matrix with the n,k entry as specfied above
        sqDmat = calcSqDistances(X, Kmus)

        #given the matrix of squared distances, determine the closest cluster
        #center for each data vector
        def determineRnk(mat):
            Rnk = np.zeros((mat.shape))
            for i in range(len(mat)):
                m = np.argmin(mat[i])
                Rnk[i, m] = 1
            return Rnk
        #R is the "responsibility" matrix
        #R will be an N-by-K matrix of binary values whose n,k entry is set as
        #per Bishop (9.2)
        #Specifically, the n,k entry is 1 if point n is closest to cluster k,
        #and is 0 otherwise
        Rnk = determineRnk(sqDmat)

        KmusOld = Kmus
        plotCurrent(X, Rnk, Kmus)
        time.sleep(1)

        #recalculate mu values based on cluster assignments as per Bishop (9.4)
        def recalcMus(X, Rnk):
            N = Rnk.shape[1]
            K = X.shape[1]
            X_new = np.zeros((N,K))
            for i in range(N):
                X_new[i, :] = (np.sum(np.asarray([Rnk[:,i], Rnk[:,i]]).T * X, axis = 0) / (np.sum(Rnk[:,i], axis = 0)))
            return X_new
        Kmus = recalcMus(X, Rnk)





        #check to see if the cluster centers have converged.  If so, break.
        if np.sum(np.abs(KmusOld.reshape((-1, 1)) - Kmus.reshape((-1, 1)))) < 1e-6:
            print(iter)
            break

    plotCurrent(X, Rnk, Kmus)
