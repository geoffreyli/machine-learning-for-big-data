import os
from pyspark import SparkConf, SparkContext


# Set custom amount of memory/cores to use
cores = '\'local[*]\''
memory = '10g'
pyspark_submit_args = ' --master ' + cores + ' --driver-memory ' + memory + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

conf = SparkConf()
sc = SparkContext(conf=conf)


# Create RDD for data points
edges = sc\
    .textFile('./q2/data/graph-full.txt', 4)\
    .map(lambda x: [int(y) for y in x.split()])

# Remove duplicate edges and treat them as the same edge
uniqueEdges = edges\
    .map(lambda x: ((x[0], x[1]), 1))\
    .reduceByKey(lambda n1, n2: n1 + n2)\
    .map(lambda x: (x[0], 1))

# Calculate number of outlinks (degree) of each node
nodeInvDegrees = uniqueEdges\
    .map(lambda x: (x[0][0], 1))\
    .reduceByKey(lambda n1, n2: n1 + n2)\
    .map(lambda x: (x[0], 1/x[1]))


# Performs dot product between one row of M and the r vector
def perform_dot(iterable):
    res = 1
    for inner_iterable in iterable:
        for num in inner_iterable:
            res *= num
    return res

# Performs update to each value of PageRank vector one at a time
def update_PageRank(node_id, edge, deg, pr, b, n):
    originNodes = edge.filter(lambda x: x[0][1] == node_id).map(lambda x: x[0][0]).collect()
    deg_filtered = deg.filter(lambda x: x[0] in originNodes)
    pr_filtered = sc.parallelize([(k, v) for k, v in pageRank.items() if k in originNodes])

    return sum(deg_filtered.cogroup(pr_filtered)\
               .mapValues(perform_dot)\
               .map(lambda x: x[1]).collect())*b + (1-b)/n


# Initialize PageRank vector estimate R
pageRank = nodeInvDegrees.map(lambda x: (x[0], 1)).collectAsMap()

# Perform PageRank update iterations
beta = 0.8
num_nodes = len(nodeInvDegrees.collect())

for k in range(40):
    for i in range(1, num_nodes+1):
        pageRank[i] = update_PageRank(i, uniqueEdges, nodeInvDegrees, pageRank, beta, num_nodes)


with open('./q2/PageRank-top5-nodes-full.txt', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in sorted(pageRank.items(), key=lambda x: x[1], reverse=True)[0:5]))

with open('./q2/PageRank-bot5-nodes-full.txt', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in sorted(pageRank.items(), key=lambda x: x[1], reverse=True)[-5:]))


sc.stop()




