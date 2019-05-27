import numpy as np
import sys

features = np.loadtxt('./q1/data/features.txt', delimiter=',')
target = np.loadtxt('./q1/data/target.txt', delimiter=',')


def computecost(w, b, x, y, C):
    loss = 0
    for i in range(len(y)):
        loss += max([0, 1-y[i]*(np.dot(w, x[i])+b)])
    return 0.5*np.sum(w**2) + C*loss

def computegrad_wj(w, b, x, y, C, j, start, stop):
    loss = 0
    for i in range(start, stop):
        if np.dot(w, x[i, :]) + b >= 1:
            loss += 0
        else:
            loss += -y[i]*x[i, j]
    return w[j] + C*loss

def computegrad_b(w, b, x, y, C, start, stop):
    loss = 0
    for i in range(start, stop):
        if np.dot(w, x[i, :]) + b >= 1:
            loss += 0
        else:
            loss += -y[i]
    return C * loss


def batch_gd(eta, conv_crit, x, y, C):
    w = np.zeros(x.shape[1])
    b = 0
    k = 0
    delta_perccost = sys.maxsize
    obj_list = [computecost(w, b, x, y, C)]

    while delta_perccost >= conv_crit:
        w_last = w.copy()
        b_last = b

        for j in range(len(w)):
            w[j] = w_last[j] - eta*computegrad_wj(w_last, b_last, x, y, C, j, 0, len(y))
        b = b_last - eta*computegrad_b(w_last, b_last, x, y, C, 0, len(y))
        k += 1
        obj_list.append(computecost(w, b, x, y, C))
        delta_perccost = np.abs(obj_list[k-1] - obj_list[k])/obj_list[k-1]*100

    return obj_list

# Instantiate and run Batch Gradient Descent
C = 100
eta_bgd = 3*10**-7
eps_bgd = 0.25
obj_bgd = batch_gd(eta_bgd, eps_bgd, features, target, C)



def stochastic_gd():