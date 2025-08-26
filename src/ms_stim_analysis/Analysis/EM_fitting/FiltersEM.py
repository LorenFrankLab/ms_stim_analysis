from pylab import *
import numpy as np
from operator import truediv

"""
------------------------------------------------------------------------------
convert to a prob
------------------------------------------------------------------------------
"""


def TransformToProb(meanv, sigma2, mu):
    # compute upper and lower bounds of conf intervals by simulation
    NUM_SAMPS = 10000

    np.random.seed(0)  # start at same random number each time
    T = len(meanv)
    p = np.zeros(T)
    pll = np.zeros(T)
    pul = np.zeros(T)
    pmode = np.zeros(T)
    sigma = np.sqrt(sigma2)

    for t in range(T):
        s = np.random.normal(meanv[t], sigma[t], NUM_SAMPS)
        ps = map(truediv, np.exp(s + mu), (1.0 + np.exp(s + mu)))
        ps = list(ps)  # JAG ADDED
        sorted_ps = sorted(ps)
        pmode[t] = np.exp(meanv[t] + mu) / (1.0 + np.exp(meanv[t] + mu))
        p[t] = np.mean(list(ps))
        pll[t] = sorted_ps[int(0.05 * NUM_SAMPS) - 1]
        pul[t] = sorted_ps[int(0.95 * NUM_SAMPS)]

    return (pmode, p, pll, pul)


"""
------------------------------------------------------------------------------
NEWTONS METHOD FOR FORWARD FILTER
------------------------------------------------------------------------------
"""


def NewtonSolve(x_prior, sigma_prior, N, Nmax, mu):

    xp = x_prior
    sp = sigma_prior

    it = xp + sp * (
        N - Nmax * np.exp(mu + xp) / (1.0 + np.exp(mu + xp))
    )  # starting iteration

    for i in range(30):
        g = xp + sp * (N - Nmax * np.exp(mu + it) / (1.0 + np.exp(mu + it))) - it
        gprime = -Nmax * sp * np.exp(mu + it) / (1.0 + np.exp(mu + it)) ** 2.0 - 1.0
        x = it - g / gprime

        if np.abs(x - it) < 1e-10:
            # print 'cvged'
            return x
        it = x

    # if no value found try different ICs: needed if there are consec same values
    it = -1
    for i in range(30):
        g = xp + sp * (N - Nmax * np.exp(mu + it) / (1.0 + np.exp(mu + it))) - it
        gprime = -Nmax * sp * np.exp(mu + it) / (1.0 + np.exp(mu + it)) ** 2.0 - 1.0
        x = it - g / gprime

        if np.abs(x - it) < 1e-10:
            # print 'cvged in 2nd'
            return x
        it = x

    # if no value found try different ICs
    it = 1
    for i in range(30):
        g = xp + sp * (N - Nmax * np.exp(mu + it) / (1 + np.exp(mu + it))) - it
        gprime = -Nmax * sp * np.exp(mu + it) / (1 + np.exp(mu + it)) ** 2 - 1.0
        x = it - g / gprime

        if np.abs(x - it) < 1e-10:
            # print 'cvged'
            return x
        it = x


"""
------------------------------------------------------------------------------
BACKWARD FILTER
------------------------------------------------------------------------------
"""


def BackwardFilter(x_post, x_prior, sigma2_post, sigma2_prior):

    T = len(x_post)
    # Initial conditions
    x_T = zeros(T)
    x_T[T - 1] = x_post[T - 1]
    sigma2_T = zeros(T)
    sigma2_T[T - 1] = sigma2_post[T - 1]
    A = np.zeros(T)

    for t in range(T - 2, 0, -1):
        A[t] = sigma2_post[t] / sigma2_prior[t + 1]
        x_T[t] = x_post[t] + np.dot(A[t], x_T[t + 1] - x_prior[t + 1])
        Asq = np.dot(A[t], A[t])
        diff_v = sigma2_T[t + 1] - sigma2_prior[t + 1]
        sigma2_T[t] = sigma2_post[t] + np.dot(Asq, diff_v)

    return x_T, sigma2_T, A


"""
--------------------------------------------------------------------
EM METHOD
--------------------------------------------------------------------
"""


def EM(xx, mu, sigma2e, x_init, sigma_init):

    num_its = range(0, 3000)
    savesigma2_e = np.zeros(len(num_its) + 1)
    savesigma2_e[0] = sigma2e

    # run though until convergence
    its = 0
    diff_its = 1
    max_its = 3000

    while diff_its > 0.00001 and its <= max_its:

        its += 1

        x_prior, x_post, sigma2_prior, sigma2_post, ape = FwdFilterEM(
            xx.values, 1, x_init, sigma_init, sigma2e, mu
        )

        x_T, sigma2_T, A = BackwardFilter(x_post, x_prior, sigma2_post, sigma2_prior)

        x_T[0] = 0
        sigma2_T[0] = sigma2e

        sigma2e = MSTEP(x_T, sigma2_T, A)

        savesigma2_e[its + 1] = sigma2e
        diff_its = abs(savesigma2_e[its + 1] - savesigma2_e[its])

        x_init = 0
        sigma_init = sigma2_T[0]

    if its == max_its:
        converge_flag = 1
        print("Did not converge in 3000 iterations")
    else:
        converge_flag = 0
        print("Converged after " + str(its) + " iterations")
        print("sigma2e is ", sigma2e)

    x_post = x_post[1:]
    sigma2_post = sigma2_post[1:]
    return x_T[1:], sigma2_T, sigma2e, sigma_init, converge_flag
    #   return x_post,sigma2_post,sigma2e,sigma_init,converge_flag


"""
--------------------------------------------------------------------
MSTEP OF EM
-------
"""


def MSTEP(xnew, signewsq, A):

    T = len(xnew)
    xnewt = xnew[2:T]
    xnewtm1 = xnew[1 : T - 1]
    signewsqt = signewsq[2:T]
    A = A[1 : T - 1]
    covcalc = np.multiply(signewsqt, A)

    term1 = np.dot(xnewt, xnewt) + np.sum(signewsqt)
    term2 = np.sum(covcalc) + np.dot(xnewt, xnewtm1)

    term3 = 2 * xnew[1] * xnew[1] + 2 * signewsq[1]
    term4 = xnew[T - 1] ** 2 + signewsq[T - 1]

    newsigsq = (2 * (term1 - term2) + term3 - term4) / T

    return newsigsq


"""
------------------------------------------------------------------------------
FORWARD FILTER SAME AS IN SMITH ET AL. 2004
------------------------------------------------------------------------------
"""


def FwdFilterEM(y, delta, x_init, sigma2_init, sigma2e, mu):

    T = len(y)

    # Data structures
    x_prior = zeros(T + 1)
    x_post = zeros(T + 1)
    sd1 = zeros(T + 1)

    next_pred_error = zeros(T + 1)

    sigma2_prior = zeros(T + 1)
    sigma2_post = zeros(T + 1)

    # FORWARD FILTER
    x_post[0] = x_init
    sigma2_post[0] = sigma2_init

    for t in range(1, T + 1):

        x_prior[t] = x_post[t - 1]
        sigma2_prior[t] = sigma2_post[t - 1] + sigma2e

        x_post[t] = NewtonSolve(x_prior[t], sigma2_prior[t], y[t - 1], 1, mu)
        # x_post[t]  = x_prior[t] + sigma2_post[t]*(y[t] - pt)

        pt = exp(mu + x_post[t]) / (1.0 + exp(mu + x_post[t]))

        sigma2_post[t] = 1.0 / (1.0 / sigma2_prior[t] + pt * (1 - pt))

        sd1[t] = np.sqrt(sigma2_post[t])

        # next_pred_error[t] = -2*(y[t]*np.log(pt) + (1-y[t])*np.log(1-pt))
        # next_pred_error[t] = (pt-y[t])**2.0
        # allv_logs[t] = -2*next_pred_error
        # print t, x_post[t], sigma2_post[t] , y[t-1]

    ape = 0  # next_pred_error.mean()

    # allv_logs.mean()
    return x_prior, x_post, sigma2_prior, sigma2_post, ape


"""
------------------------------------------------------------------------------
END OF CODE
------------------------------------------------------------------------------
"""
