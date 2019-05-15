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


# Scale the hubbiness or authority vectors (which are represented as values of a Dict)
def scale_vector(vec):
    max_val = max(vec.values())
    return {k:v/max_val for k, v in vec.items()}


# Performs update to each value of Hubbiness vector one at a time
# In effect doing matrix multiplication of L @ a
def update_hub(node_id, edge, auth, lam):
    destNodes = edge.filter(lambda x: x[0][0] == node_id).map(lambda x: x[0][1]).collect()
    auth_filtered = [v for k, v in auth.items() if k in destNodes]
    return lam*sum(auth_filtered)


# Performs update to each value of Authority vector one at a time
# In effect doing matrix multiplication of L.T @ h
def update_auth(node_id, edge, hub, m):
    originNodes = edge.filter(lambda x: x[0][1] == node_id).map(lambda x: x[0][0]).collect()
    hub_filtered = [v for k, v in hub.items() if k in originNodes]
    return m*sum(hub_filtered)


# Initialize Hubbiness and Authority vectors
h = uniqueEdges.map(lambda x: (x[0][0], 1)).reduceByKey(lambda n1, n2: 1).collectAsMap()
a = dict()

# Perform Hub and Authority update iterations
num_nodes = len(h)
lamb = 1
mu = 1
iter = 40

for k in range(iter):
    for i in range(1, num_nodes+1):
        a[i] = update_auth(i, uniqueEdges, h, mu)
    a = scale_vector(a)

    for i in range(1, num_nodes+1):
        h[i] = update_hub(i, uniqueEdges, a, lamb)
    h = scale_vector(h)


# Write Output to File
with open('./q2/HITS-full-hub-top5-nodes.txt', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in sorted(h.items(), key=lambda x: x[1], reverse=True)[0:5]))

with open('./q2/HITS-full-hub-bot5-nodes.txt', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in sorted(h.items(), key=lambda x: x[1], reverse=True)[-5:]))

with open('./q2/HITS-full-auth-top5-nodes.txt', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in sorted(a.items(), key=lambda x: x[1], reverse=True)[0:5]))

with open('./q2/HITS-full-auth-bot5-nodes.txt', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in sorted(a.items(), key=lambda x: x[1], reverse=True)[-5:]))


