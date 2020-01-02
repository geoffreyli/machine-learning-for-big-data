import numpy as np
import pandas as pd

r = np.loadtxt('./q4/data/user-shows.txt', dtype=int)
shows = pd.Series(pd.read_csv('./q4/data/shows.txt', header=None)[0])

# Part (d.1) Compute P and Q

p = np.zeros(shape=(r.shape[0], r.shape[0]), dtype='int')
q = np.zeros(shape=(r.shape[1], r.shape[1]), dtype='int')

num_users_per_show = np.sum(r, axis=0)
num_shows_per_user = np.sum(r, axis=1)

for m in range(r.shape[0]):
    p[m][m] = num_shows_per_user[m]

for n in range(r.shape[1]):
    q[n][n] = num_users_per_show[n]


# Part (d.2-3) Compute gamma for user-user collaborative filtering

# Computing element-wise inverse square root of P and Q
p_invsqrt = np.reciprocal(p**0.5, where=p!=0)
q_invsqrt = np.reciprocal(q**0.5, where=q!=0)

# Similarity matrix for user-user and item-item
sim_uu = p_invsqrt @ r @ r.T @ p_invsqrt
sim_ii = q_invsqrt @ r.T @ r @ q_invsqrt

# Computing gamma recommendation matrix for user-user and item-item
gamma_uu = sim_uu @ r
gamma_ii = r @ sim_ii


# Find show recommendations
s = range(0,100)
alex_idx = 499

# Top 5 show recommendations for Alex for user-user collaborative filtering
recs_uu = pd.DataFrame({
    'Show': list(shows[np.argsort(-gamma_uu[alex_idx, s])[0:5]]),
    'SimilarityScore' :list(-np.sort(-gamma_uu[alex_idx, s])[0:5])
 })

# Top 5 show recommendations for Alex for item-item collaborative filtering
recs_ii = pd.DataFrame({
    'Show': list(shows[np.argsort(-gamma_ii[alex_idx, s])[0:5]]),
    'SimilarityScore' :list(-np.sort(-gamma_ii[alex_idx, s])[0:5])
 })

recs_uu.to_csv('./q4/alex_recs_uu.txt', index=False)
recs_ii.to_csv('./q4/alex_recs_ii.txt', index=False)