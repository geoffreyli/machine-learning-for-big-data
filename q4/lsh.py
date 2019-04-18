# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import numpy as np
import random
import timeit
import pdb
import unittest
from PIL import Image
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    return np.sum(np.abs(u - v))

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    A_wo_query = filter(lambda i: i != query_index, range(len(A)))

    distances = map(lambda r: (r, l1(A[r], A[query_index])), A_wo_query)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]


# TODO: Write a function that computes the error measure
def compute_error(A, query_indices, nn_linear, nn_lsh):
    error = list()

    for query_index in query_indices:
        lsh_distances = map(lambda r: l1(A[r], A[query_index]), nn_lsh[query_index])
        linear_distances = map(lambda r: l1(A[r], A[query_index]), nn_linear[query_index])
        error.append(sum(lsh_distances)/sum(linear_distances))

    return sum(error)/len(error)


# TODO: Solve Problem 4
def problem4():
    patches = load_data('./q4/data/patches.csv')
    queryPatches = [100*i-1 for i in range(1, 11)]

    # Part (d.1)

    topN_q4d1 = 3

    hashFuncs, hashedPatches = lsh_setup(patches)
    timeLinear = list()
    timeLSH = list()
    nn3Linear = dict()
    nn3LSH = dict()

    for queryIndex in queryPatches:

        start_time = timeit.default_timer()
        top3 = linear_search(patches, queryIndex, topN_q4d1)
        end_time = timeit.default_timer()
        nn3Linear[queryIndex] = top3
        timeLinear.append((end_time - start_time))

        start_time = timeit.default_timer()
        top3 = lsh_search(patches, hashedPatches, hashFuncs, queryIndex, topN_q4d1)
        end_time = timeit.default_timer()
        nn3LSH[queryIndex] = top3
        timeLSH.append((end_time - start_time))


    timeLinear_mean = np.mean(timeLinear)
    timeLSH_mean = np.mean(timeLSH)

    print('Linear Search')
    print(nn3Linear)
    print(timeLinear_mean)
    print('')

    print('LSH Search')
    print(nn3LSH)
    print(timeLSH_mean)
    print('')

    print("Part d.1 complete")
    print("")

    # Part (d.2)

    k_set = 24
    L_var = [10, 12, 14, 16, 18, 20]
    L_set = 10
    k_var = [16, 18, 20, 22, 24]
    topN_q4d2 = 3

    l_plot_errors = list()

    for l_instance in L_var:
        hashFuncs_l, hashedPatches_l = lsh_setup(patches, k=k_set, L=l_instance)
        nn3Linear_l = dict()
        nn3LSH_l = dict()

        for queryIndex in queryPatches:
            nn3Linear_l[queryIndex] = linear_search(patches, queryIndex, topN_q4d2)
            nn3LSH_l[queryIndex] = lsh_search(patches, hashedPatches_l, hashFuncs_l, queryIndex, topN_q4d2)

        l_plot_errors.append(compute_error(patches, queryPatches, nn3Linear_l, nn3LSH_l))

    plt.figure(0)
    plt.plot(L_var, l_plot_errors)
    plt.savefig('l_plot.png')
    plt.close('l_plot.png')


    k_plot_errors = list()

    for k_instance in k_var:
        hashFuncs_k, hashedPatches_k = lsh_setup(patches, k=k_instance, L=L_set)
        nn3Linear_k = dict()
        nn3LSH_k = dict()

        for queryIndex in queryPatches:
            nn3Linear_k[queryIndex] = linear_search(patches, queryIndex, topN_q4d2)
            nn3LSH_k[queryIndex] = lsh_search(patches, hashedPatches_k, hashFuncs_k, queryIndex, topN_q4d2)

        k_plot_errors.append(compute_error(patches, queryPatches, nn3Linear_k, nn3LSH_k))

    plt.figure(1)
    plt.plot(k_var, k_plot_errors)
    plt.savefig('k_plot.png')
    plt.close('k_plot.png')

    print("Part d.2 complete")
    print("")

    # Part (d.3)

    queryIndex_q4d3 = 99
    topN_q4d3 = 10
    hashFuncs, hashedPatches = lsh_setup(patches)

    nn10Linear = linear_search(patches, queryIndex_q4d3, topN_q4d3)
    plot(patches, nn10Linear + [queryIndex_q4d3], 'nn10Linear')
    print("nn10Linear")
    print(nn10Linear)
    print("")

    nn10LSH = lsh_search(patches, hashedPatches, hashFuncs, queryIndex_q4d3, topN_q4d3)
    plot(patches, nn10LSH + [queryIndex_q4d3], 'nn10LSH')
    print("nn10LSH")
    print(nn10LSH)
    print("")

    print("Part d.3 complete")
    print("")


#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


if __name__ == '__main__':
#    unittest.main() ### TODO: Uncomment this to run tests
    problem4()
