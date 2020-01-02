import numpy as np
import matplotlib.pyplot as plt


# Performs update to P and Q per line of training data read per SGD update formulas
# If the particular movie/user ID has not been seen yet, initialize with rand floats
def update_round(u, i, r, eta):
    if u not in p.keys():
        p[u] = np.random.uniform(0, (5/k)**0.5, size=(k))
    if i not in q.keys():
        q[i] = np.random.uniform(0, (5/k)**0.5, size=(k))

    q_old = q[i]
    p_old = p[u]
    eps = 2*(r - np.dot(q_old, p_old.T))

    q[i] = q_old + eta*(eps*p_old - 2*lamb*q_old)
    p[u] = p_old + eta*(eps*q_old - 2*lamb*p_old)


# Computes empirical risk error at the end of the iteration
# (missing the P and Q norms, those are added later)
def compute_error(u, i, r):
    return (r - np.dot(q[i], p[u].T))**2


# Initialize variables
k = 20
lamb = 0.1
max_iter = 40

total_error = dict()


# Performs iterative SGD for input eta (learning rate)
def sgd(eta):

    for i in range(max_iter):

        with open('./q3/data/ratings.train.txt') as f:
            for line in f:
                line_u, line_i, line_r = line.split()
                update_round(line_u, line_i, float(line_r), eta)

        iter_error = 0
        with open('./q3/data/ratings.train.txt') as f:
            for line in f:
                line_u, line_i, line_r = line.split()
                iter_error += compute_error(line_u, line_i, float(line_r))

        p_norms = 0
        for n in p.keys():
            p_norms += lamb * (np.sum(p[n] ** 2))

        q_norms = 0
        for m in q.keys():
            q_norms += lamb * (np.sum(q[m] ** 2))

        iter_error += p_norms + q_norms

        total_error[eta].append(iter_error)


# Find best learning rate
converge_targ = 65000
opt_eta = None

for ss in [i/100 for i in range(5, 0, -1)]:
    q = dict()
    p = dict()
    total_error[ss] = list()

    sgd(ss)
    if total_error[ss][len(total_error[ss])-1] <= converge_targ:
        opt_eta = ss
        break

print(opt_eta)

plt.figure(0)
plt.plot(list(range(1, len(total_error[opt_eta])+1)), total_error[opt_eta])
plt.title('Objective Function E vs. Iterations (learning rate = '+str(opt_eta)+')')
plt.xlabel('Iteration')
plt.ylabel('Objective Function E')
plt.savefig('./q3/objfunc-iterations.png')
plt.close(0)