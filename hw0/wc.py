import re
import sys
from pyspark import SparkConf, SparkContext

# Create spark context
conf = SparkConf()
sc = SparkContext(conf=conf)

# Read in target file into an RDD
lines = sc.textFile(sys.argv[1])

# Split the lines into individual words
words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))

# Replace each word with a tuple consisting of that word and 1
pairs = words.map(lambda w: (w, 1))

# Group the pairs RDD by key (word) and add up the values
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)

# Store the results in a file and stop the context
counts.saveAsTextFile(sys.argv[2])
sc.stop()

