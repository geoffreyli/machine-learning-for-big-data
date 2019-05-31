import numpy as np
import sys
from timeit import default_timer
import matplotlib.pyplot as plt

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
        if y[i]*(np.dot(w, x[i, :])+b) >= 1:
            loss += 0
        else:
            loss += -y[i]*x[i, j]
    return w[j] + C*loss

def computegrad_b(w, b, x, y, C, start, stop):
    loss = 0
    for i in range(start, stop):
        if y[i]*(np.dot(w, x[i, :])+b) >= 1:
            loss += 0
        else:
            loss += -y[i]
    return C * loss


def batch_gd(eta, conv_crit, x, y, C):
    print('Starting Batch Gradient Descent...')
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

start = default_timer()
obj_bgd = batch_gd(eta_bgd, eps_bgd, features, target, C)
end = default_timer()
convg_time_bgd = end - start
print(convg_time_bgd)



def stochastic_gd(eta, conv_crit, x, y, C):
    print('Starting Stochastic Gradient Descent...')
    w = np.zeros(x.shape[1])
    b = 0
    k = 0
    i = 1
    delta_perccost_k = sys.maxsize
    obj_list = [computecost(w, b, x, y, C)]

    while delta_perccost_k >= conv_crit:
        w_last = w.copy()
        b_last = b
        if k == 0:
            delta_perccost_k = 0

        for j in range(len(w)):
            w[j] = w_last[j] - eta * computegrad_wj(w_last, b_last, x, y, C, j, i-1, i)
        b = b_last - eta * computegrad_b(w_last, b_last, x, y, C, i-1, i)

        k += 1
        i = (i % len(y)) + 1
        obj_list.append(computecost(w, b, x, y, C))

        delta_perccost_last = delta_perccost_k
        delta_perccost_curr = np.abs(obj_list[k-1] - obj_list[k])/obj_list[k-1]*100
        delta_perccost_k = 0.5*delta_perccost_last + 0.5*delta_perccost_curr

    return obj_list


# Instantiate and run Stochastic Gradient Descent
shuffled_indices = np.random.permutation(range(len(features)))
C = 100
eta_sgd = 0.0001
eps_sgd = 0.001

start = default_timer()
obj_sgd = stochastic_gd(eta_sgd, eps_sgd, features[shuffled_indices, :], target[shuffled_indices], C)
end = default_timer()
convg_time_sgd = end - start
print(convg_time_sgd)


def minibatch_gd(eta, B, conv_crit, x, y, C):
    print('Starting Mini Batch Gradient Descent...')
    w = np.zeros(x.shape[1])
    b = 0
    k = 0
    l = 0
    delta_perccost_k = sys.maxsize
    obj_list = [computecost(w, b, x, y, C)]

    while delta_perccost_k >= conv_crit:
        w_last = w.copy()
        b_last = b
        if k == 0:
            delta_perccost_k = 0

        for j in range(len(w)):
            w[j] = w_last[j] - eta * computegrad_wj(w_last, b_last, x, y, C, j, l*B, min(len(y), (l+1)*B))
        b = b_last - eta * computegrad_b(w_last, b_last, x, y, C, l*B, min(len(y), (l+1)*B))

        k += 1
        l = np.int((l+1 % np.ceil(len(y)/B)) + 1)
        obj_list.append(computecost(w, b, x, y, C))

        delta_perccost_last = delta_perccost_k
        delta_perccost_curr = np.abs(obj_list[k - 1] - obj_list[k]) / obj_list[k - 1] * 100
        delta_perccost_k = 0.5 * delta_perccost_last + 0.5 * delta_perccost_curr

    return obj_list


# Instantiate and run Mini-Batch Gradient Descent
shuffled_indices_mbgd = np.random.permutation(range(len(features)))
C = 100
eta_mbgd = 10**-5
eps_mbgd = 0.01
B = 20

start = default_timer()
obj_mbgd = minibatch_gd(eta_mbgd, B, eps_mbgd, features[shuffled_indices_mbgd, :], target[shuffled_indices_mbgd], C)
end = default_timer()
convg_time_mbdg = end - start
print(convg_time_mbdg)


# Plots of iterations vs. cost function
plt.figure()
plt.plot(range(len(obj_bgd)), obj_bgd)
plt.plot(range(len(obj_sgd)), obj_sgd)
plt.plot(range(len(obj_mbgd)), obj_mbgd)
plt.legend(['Batch GD', 'Stochastic GD', 'Mini-Batch GD'], loc='upper right')
plt.title('Cost Function vs. Iteration t')
plt.xlabel('Iteration t')
plt.ylabel('Cost Function Value')
plt.savefig('./q1/cost-func-vs-iterations.png');



