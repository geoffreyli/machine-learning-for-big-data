import re
import sys
from pyspark import SparkConf, SparkContext

print("Hello World!")

# Create spark context
conf = SparkConf()
sc = SparkContext(conf=conf)

# Read in target file into an RDD
lines = sc.textFile(sys.argv[1])

# Split the lines into individual words
words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))

# Convert all words to lower case and extract first letter
lower_words = words.map(lambda w: w.lower())

# Filter words to have length of at least 1 (remove empty space "words")
lower_words_nonempty = lower_words.filter(lambda lw: len(lw) > 0)

# Extract just the first letter of the word
first_char = lower_words_nonempty.map(lambda lwn: lwn[0])

# Filter out non-alphabetic characters
first_letter = first_char.filter(lambda fc: fc.isalpha())

# Replace each word with a tuple consisting of that word and 1
pairs = first_letter.map(lambda w: (w, 1))

# Group the pairs RDD by key (word) and add up the values
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2).sortByKey()

# Store the results in a file and stop the context
counts.saveAsTextFile(sys.argv[2])
sc.stop()


# script for concatenating output files into one file
# cat part-00000 part-00001 > output.txt