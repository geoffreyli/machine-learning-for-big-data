import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext

# Set custom amount of memory/cores to use
cores = '\'local[*]\''
memory = '10g'
pyspark_submit_args = ' --master ' + cores + ' --driver-memory ' + memory + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

conf = SparkConf()
sc = SparkContext(conf=conf)


# Create RDD for data points
data = sc\
    .textFile('./q2/data/data.txt', 4)\
    .map(lambda x: np.array([float(y) for y in x.split()]))


# Create RDDs for c1 and c2 and map each centroid to an index, to act as an ID
# We can CollectAsMap because we can expect the # of centroids to be small (10 in this case)
c1 = sc.textFile('./q2/data/c1.txt')\
    .map(lambda x: [float(y) for y in x.split()]) \
    .zipWithIndex()\
    .map(lambda l: (l[1], np.array(l[0]))).collectAsMap()

c2 = sc.textFile('./q2/data/c2.txt')\
    .map(lambda x: [float(y) for y in x.split()]) \
    .zipWithIndex()\
    .map(lambda l: (l[1], np.array(l[0]))).collectAsMap()


# mapCentroid
#   input: point coordinate array, euclidean or manhattan distance, centroid dict
#   output: (K, V) where
#       K is the ID of the closest centroid and
#       V is (point coordinate array, cost function for this point, 1)

def mapCentroid(p, dist_type, c):
    min_dist = sys.float_info.max
    min_clust = None
    cost = None

    for i in c.keys():
        if dist_type == 'Euclidean':
            dist = np.sqrt(np.sum((p-c[i])**2))  # Euclidean distance
        elif dist_type == 'Manhattan':
            dist = np.sum(np.abs((p-c[i])))  # Manhattan distance

        if dist < min_dist:
            min_dist = dist
            min_clust = i

    cost = min_dist  # cost is min_dist for both types of distance measures

    return (min_clust, (p, cost, 1))


# reduceCentroid
#   input: mapped centroid RDD
#   output: aggregate statistics by summing the values in tuple

def reduceCentroid(cd1, cd2):
    coord_sum = cd1[0] + cd2[0]
    cost_sum = cd1[1] + cd2[1]
    n_sum = cd1[2] + cd2[2]
    return (coord_sum, cost_sum, n_sum)



# RUN ITERATIVE K-MEANS
# Output: plots of total costs vs. iterations
max_iter = 20
centroid_inits = ['C1', 'C2']
dist_metrics = ['Euclidean', 'Manhattan']
improvement_idx = 9

for dist_metric in dist_metrics:

    for ci in centroid_inits:

        if ci == 'C1':
            centroid = c1
        elif ci == 'C2':
            centroid = c2

        cost_by_iteration = list()

        for i in range(0, max_iter):
            # Map points to clusters
            map_points = data.map(lambda p: mapCentroid(p, dist_metric, centroid))

            # Summarize clusters by adding
            cluster_summ = map_points.reduceByKey(reduceCentroid)

            # Compute new clusters by finding mean point of each cluster; collect as dict and update
            centroid = cluster_summ.map(lambda cs: (cs[0], cs[1][0] / cs[1][2])).collectAsMap()

            # Compute total cost by cluster and keep track for plotting
            total_cost = cluster_summ.map(lambda cs: cs[1][1]).sum()
            cost_by_iteration.append(total_cost)

        plt.figure(ci+dist_metric)
        plt.plot(list(range(len(cost_by_iteration))), cost_by_iteration)
        plt.title('Total Cost vs. Iteration ('+ci+' Centroid Init; '+dist_metric+' Distance)')
        plt.xlabel('Iteration')
        plt.ylabel('Total Cost')
        plt.savefig('./q2/'+ci+'-'+dist_metric+'_costs.png')
        plt.close(ci+dist_metric)

        print('('+dist_metric+' Distance; '+ci+
              ' Centroid Initialization): Drop in cost after 10 iterations = '+
              str(cost_by_iteration[improvement_idx]/cost_by_iteration[0]-1))


sc.stop()



