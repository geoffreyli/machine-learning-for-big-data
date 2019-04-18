import re
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
print("lines")
print(lines.take(10))

# Create RDD with (Users, [list of friends])
# Convert userID to integers, since we'll need to sort numerically later
# friends = lines.map(lambda u: (u[0], u[1].split(',')))


def cleanLines(l):
    if l[1].strip():
        return (int(l[0]), list(map(int, l[1].split(','))))
    else:
        return (int(l[0]), [])


friends = lines.map(cleanLines)
print("friends")
print(friends.take(10))


# Create set of all users
allUsers = friends.keys().collect()
print("allUsers")
print(type(allUsers))
print(len(allUsers))

# Create set of (Users, [list of non-friends])
# non-friend list is for set of possible friend recommendations, since you only want to recommend non-friends
# confirmed this created the correct set of non-friends
# nonfriends = friends.map(lambda x: (x[0], list(set(allusers) - set(x[1]))))



# Create RDD for all pairs of friends for each user's friend list
# For each user, create a RDD where:
# KEY = (Friend1, Friend2) across all combinations of pairs in that user's friend list
# VALUE = 1
friendPairsByUser = friends\
    .map(lambda x: [(x[1][i],x[1][j]) for i in range(len(x[1])) for j in range(i+1, len(x[1]))])\
    .flatMap(lambda x: x)\
    .map(lambda x: (x, 1))
print("friendPairsByUser")
print(friendPairsByUser.take(5))

friendPairsByUser = friendPairsByUser.map(lambda x: ((sorted(x[0])[0], sorted(x[0])[1]), 1))

# Now Reduce by grouping on the (Friend1,Friend2) pairs and summing the Values
# The sum of the Values should produce the number of mutual friends between that Friend pair
mutualFriendsbyFriendPair = friendPairsByUser.reduceByKey(lambda n1, n2: n1 + n2)
print("mutualFriendsbyFriendPair")
print(mutualFriendsbyFriendPair.take(5))


# Some of the (Friend1, Friend2) pairs are already friends: we need to remove these from our list
# since we wouldn't want to recommend an existing friend.

friendsDict = friends.collectAsMap()
print("friendsDict")
print(friendsDict[0])

def checkIfAlreadyFriends(x):
    user1, user2 = x[0]
    if user1 not in friendsDict[user2]:
        return True
    else:
        return False

mutualFriendsNonExistingFriendPairs = mutualFriendsbyFriendPair.filter(checkIfAlreadyFriends)
print("mutualFriendsNonExistingFriendPairs")
print(mutualFriendsNonExistingFriendPairs.take(5))


# Now we need to find the recommended friend lists for each user.
# Restructure the RDD such that Friend1 becomes the KEY
# and the VALUE is (Friend2, CountOfMutualFriends)

# friendRecs_part1 = mutualFriendsNonExistingFriendPairs.map(lambda x: (x[0][0], (x[0][1], x[1])))
# friendRecs_part2 = mutualFriendsNonExistingFriendPairs.map(lambda x: (x[0][1], (x[0][0], x[1])))

friendRecs = mutualFriendsNonExistingFriendPairs.map(lambda x: (x[0][0], (x[0][1], x[1])))\
    .union(mutualFriendsNonExistingFriendPairs.map(lambda x: (x[0][1], (x[0][0], x[1]))))\
    .sortByKey()

print("friendRecs")
print(friendRecs.take(10))

# Map the friendRecs RDD value to a list of tuples (so we can reduceByKey and append list of tuples)
# ReduceByKey to combine all friend recommendations by userID by appending to list.
# End result RDD: (UserID, [(FriendRec1, #mutualfriends), (FriendRec2, #mutualfriends), ... ])
friendRecsByUser = friendRecs.map(lambda x: (x[0], [ x[1] ])).reduceByKey(lambda fr1, fr2: fr1 + fr2)
print("friendRecsByUser")
print(friendRecsByUser.take(5))

# Now we sort the friendRecsByUser RDD first by descending number of mutual friends, then ascending userID
friendRecsByUserSorted = friendRecsByUser.map(lambda x: (x[0], sorted(x[1], key=lambda y: (-y[1], y[0]))))
print("friendRecsByUserSorted")
print(friendRecsByUserSorted.take(5))

friendRecsByUserSorted = friendRecsByUserSorted.sortByKey()

# print(friendRecsByUserSorted.filter(lambda x: x[0] == 11).collect())
# print(friendRecsByUserSorted.filter(lambda x: x[0] == 49991).collect())




output_IDs = [924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]


output_recs = friendRecsByUserSorted.filter(lambda x: x[0] in output_IDs)

output_recs.saveAsTextFile('./q1_output')