import numpy as np
import matplotlib.pyplot as plt


# Read in hash_params and counts

hash_params = np.loadtxt('./q4/data/hash_params.txt', delimiter='\t')
# counts_tiny = np.loadtxt('./q4/data/counts_tiny.txt', delimiter='\t')
counts = np.loadtxt('./q4/data/counts.txt', delimiter='\t')


# Define functions

def hash_fun(a, b, p, n_buckets, x):
    y = x % p
    hash_val = (a*y + b) % p
    return hash_val % n_buckets


def init_count_dict(n_hash, n_buck):
    count_dict = dict()
    for i in range(n_hash):
        count_dict[i] = dict()
        for j in range(n_buck):
            count_dict[i][j] = 0
    return count_dict


def increment_count(x, h_c):
    for i in range(num_hash_fun):
        j = hash_fun(hash_params[i][0], hash_params[i][1], p, num_buckets, x)
        h_c[i][j] += 1
    return h_c


def finalize_estimate_counts(h_c, uniq_ite):
    final_estimate = dict()
    for i in uniq_ite:
        bucket_counts = list()
        for h in range(num_hash_fun):
            h_i = hash_fun(hash_params[h][0], hash_params[h][1], p, num_buckets, i)
            bucket_counts.append(h_c[h][h_i])
        final_estimate[i] = min(bucket_counts)
    return final_estimate


# Set parameters

p = 123457
delta = np.exp(-5)
sigma = np.exp(1)*10**-4

num_hash_fun = np.int(np.ceil(np.log(1/delta)))
num_buckets = np.int(np.ceil(np.exp(1)/sigma))
unique_items = [np.int(counts[i, 0]) for i in range(len(counts))]
# Note: we have unique_items in this exercise, but in practice we would need to keep
# a tally as the stream comes in


# Run the algorithm

hashed_counts = init_count_dict(num_hash_fun, num_buckets)
with open('./q4/data/words_stream.txt') as f:
    for item in f:
        hashed_counts = increment_count(int(item), hashed_counts)


estimated_counts = finalize_estimate_counts(hashed_counts, unique_items)

estimated_error = dict()
for item in unique_items:
    estimated_error[item] = (estimated_counts[item] - counts[item-1][1])/counts[item-1][1]

exact_word_frequency = dict()
total_count = np.sum(counts[:, 1])
for item in unique_items:
    exact_word_frequency[item] = counts[item-1][1]/total_count



# Plot the output

plt.figure()
plt.xscale('log')
plt.xticks([10**i for i in range(-9, -1)])
plt.yscale('log')
plt.yticks([10**i for i in range(-5, 8)])
plt.scatter([exact_word_frequency[i] for i in unique_items],
            [estimated_error[i] for i in unique_items], s=5)
plt.title('Log-Log Plot of Relative Error vs. Exact Word Frequency')
plt.xlabel('Exact Word Frequency')
plt.ylabel('Relative Error')
plt.savefig('./q4/relative-error-vs-exact-freq_full.png');





