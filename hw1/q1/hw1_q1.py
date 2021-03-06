import sys
import os
from pyspark import SparkConf, SparkContext

# Set custom amount of memory/cores to use
cores = '\'local[*]\''
memory = '10g'
pyspark_submit_args = ' --master ' + cores + ' --driver-memory ' + memory + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc\
    .textFile('/Users/geoffreyli/geoffli @ UW/19-2-Sp-CSE-547/Homeworks/hw1/q1/data/soc-LiveJournal1Adj.txt', 4)\
    .map(lambda x: x.split('\t'))


# Create RDD with (Users, [list of friends])
# Convert userID to integers, since we'll need to sort numerically later
# Return friends as a list of friend IDs
def cleanLines(l):
    if l[1].strip():
        return (int(l[0]), list(map(int, l[1].split(','))))
    else:
        return (int(l[0]), [])


friends = lines.map(cleanLines)

# Create RDD for all pairs of friends for each user's friend list
# For each user, create a RDD where:
#   KEY = (Friend1, Friend2) across all combinations of pairs in that user's friend list
#   VALUE = 1
friendPairsByUser = friends\
    .map(lambda x: [(x[1][i], x[1][j]) for i in range(len(x[1])) for j in range(i+1, len(x[1]))])\
    .flatMap(lambda x: x)\
    .map(lambda x: (x, 1))

# Also add tuples for each (User,Friend) combination. However for these pairs, we'll use
# max negative integer as the Value. Such that when we group by these tuple pairs, the
# value will sum to an extremely negative number, which we can use to filter out pairs that are
# already friends.
userFriendPairs = friends\
    .map(lambda x: [(x[0], x[1][i]) for i in range(len(x[1]))])\
    .flatMap(lambda x: x)\
    .map(lambda x: (x, -sys.maxsize))

allPairs = friendPairsByUser.union(userFriendPairs)

# Need to sort the tuple keys so that (Friend1,Friend2) and (Friend2,Friend1) will group together
allPairsSorted = allPairs.map(lambda x: ((sorted(x[0])[0], sorted(x[0])[1]), x[1]))

# Now Reduce by grouping on the (Friend1,Friend2) tuples and summing the Values
# The sum of the Values should produce the number of mutual friends between that Friend pair
# since we created them from the 'perspective' of the User
mutualFriendsbyFriendPair = allPairsSorted.reduceByKey(lambda n1, n2: n1 + n2)

# Some of the (Friend1, Friend2) pairs are already friends: we need to remove these from our list
# since we wouldn't want to recommend an existing friend.
# These would be the pairs with a negative mutual friend count.
mutualFriendsNonExistingFriendPairs = mutualFriendsbyFriendPair.filter(lambda x: x[1] >= 0)

# Now we need to find the recommended friend lists for each user.
# Restructure the RDD such that Friend1 becomes the KEY
# and the VALUE is (Friend2, CountOfMutualFriends with Friend1)
friendRecs = mutualFriendsNonExistingFriendPairs.map(lambda x: (x[0][0], (x[0][1], x[1])))\
    .union(mutualFriendsNonExistingFriendPairs.map(lambda x: (x[0][1], (x[0][0], x[1]))))\
    .sortByKey()

# Map the friendRecs RDD value to a list of tuples (so we can reduceByKey and append list of tuples)
# ReduceByKey to combine all friend recommendations by userID by appending to list.
# End result RDD: (UserID, [(FriendRec1, #mutualfriends), (FriendRec2, #mutualfriends), ... ])
friendRecsByUser = friendRecs.map(lambda x: (x[0], [ x[1] ])).reduceByKey(lambda fr1, fr2: fr1 + fr2)


# Now we sort the friendRecsByUser RDD first by descending number of mutual friends, then ascending userID
friendRecsByUserSorted = friendRecsByUser.map(lambda x: (x[0], sorted(x[1], key=lambda y: (-y[1], y[0]))))

# Sort by UserID Key. Not really necessary but nicer to review output
friendRecsByUserSorted = friendRecsByUserSorted.sortByKey()

# Test cases. 11 from prompt; 49991 has no friends
# print(friendRecsByUserSorted.filter(lambda x: x[0] == 11).collect())
# print(friendRecsByUserSorted.filter(lambda x: x[0] == 49991).collect())

# Extract just the list of friend recommendations for each user (without the # of mutual friends)
friendRecListByUser = friendRecsByUserSorted.map(lambda x: (x[0], [i[0] for i in x[1]]))


# Create function to write output to file
def write_recs_to_file(finalRecRDD, userList, numRecs, filename):
    finalRecDict = dict()
    if userList:
        finalRecDict = finalRecRDD.filter(lambda x: x[0] in userList).collectAsMap()
    else:
        finalRecDict = finalRecRDD.collectAsMap()

    with open(filename, 'w') as outfile:
        for user, recs in finalRecDict.items():
            outlen = min(len(recs), numRecs)
            print(str(user)+'\t'+','.join(str(x) for x in recs[:outlen]), file=outfile)


# Write all output to file (uncomment below)
write_recs_to_file(finalRecRDD=friendRecListByUser, userList=[], numRecs=10, filename='hw1q1_output_full.txt')

# Write just the requested output for HW1 Q1
output_IDs = [924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]
write_recs_to_file(finalRecRDD=friendRecListByUser, userList=output_IDs, numRecs=10, filename='hw1q1_output.txt')


