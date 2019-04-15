import itertools

# Set support threshold at 100
s = 100

# Read through the raw text file and create C1 (count of 1-tuple items)
with open('./q2/data/browsing.txt', 'r') as browsing:
    line = browsing.readline()
    C1 = dict()

    while line:
        items = line.split()
        for item in items:
            C1[item] = C1.get(item, 0) + 1

        line = browsing.readline()


# Prune C1 and only leave 1-tuple items that are greater than support threshold
L1 = {k: v for k, v in C1.items() if v >= s}

# Create C2 (2-tuple items) using L1 (only frequent 1-tuple items)
C2 = dict()
for pair in itertools.combinations(list(L1.keys()), 2):
    C2[pair] = 0

# Pass through file again and count the number of occurrences of 2-tuple items
with open('./q2/data/browsing.txt', 'r') as browsing:
    line = browsing.readline()

    while line:
        items = line.split()
        for itemPair in itertools.combinations(items, 2):
            if itemPair in C2.keys():
                C2[itemPair] += 1
            elif (itemPair[1], itemPair[0]) in C2.keys():
                C2[(itemPair[1], itemPair[0])] += 1

        line = browsing.readline()

# Prune C2 and only leave 2-tuple items that are greater than support threshold
L2 = {k: v for k, v in C2.items() if v >= s}

# Calculate the confidence of A -> B and B -> A from L2 and L1 occurrences
conf_L2 = {**{k: v/L1[k[0]] for k, v in L2.items()}, **{(k[1],k[0]): v/L1[k[1]] for k, v in L2.items()}}

# Sort and get the top 5 rules for 2-tuples
conf_L2_sorted = sorted(conf_L2.items(), key=lambda kv: (-kv[1], kv[0][0]))
conf_L2_sorted[0:5]


# Create C3 (3-tuple items) using L2 (only frequent 2-tuple items)
C3 = dict()
# Inefficient implementation just considers L1
# for triplet in itertools.combinations(list(L1.keys()), 3):
#     C3[triplet] = 0

# Efficient implementation considers L2 and L1
for triplet in itertools.combinations(list(L1.keys()), 3):
    for pair_to_check in itertools.combinations(list(triplet), 2):
        if (pair_to_check not in L2.keys()) and ((pair_to_check[1],pair_to_check[0]) not in L2.keys()):
            break
    else:
        C3[triplet] = 0


# Pass through file again and count the number of occurrences of 3-tuple items
with open('./q2/data/browsing.txt', 'r') as browsing:
    line = browsing.readline()

    # Note this could be more efficient by considering unordered sets instead of ordered tuples
    # Revise later if have time
    while line:
        items = line.split()
        for itemTriplet in itertools.combinations(items, 3):
            if (itemTriplet[0], itemTriplet[1], itemTriplet[2]) in C3.keys():
                C3[(itemTriplet[0], itemTriplet[1], itemTriplet[2])] += 1
            elif (itemTriplet[0], itemTriplet[2], itemTriplet[1]) in C3.keys():
                C3[(itemTriplet[0], itemTriplet[2], itemTriplet[1])] += 1

            elif (itemTriplet[1], itemTriplet[0], itemTriplet[2]) in C3.keys():
                C3[(itemTriplet[1], itemTriplet[0], itemTriplet[2])] += 1
            elif (itemTriplet[1], itemTriplet[2], itemTriplet[0]) in C3.keys():
                C3[(itemTriplet[1], itemTriplet[2], itemTriplet[0])] += 1

            elif (itemTriplet[2], itemTriplet[1], itemTriplet[0]) in C3.keys():
                C3[(itemTriplet[2], itemTriplet[1], itemTriplet[0])] += 1
            elif (itemTriplet[2], itemTriplet[0], itemTriplet[1]) in C3.keys():
                C3[(itemTriplet[2], itemTriplet[0], itemTriplet[1])] += 1

        line = browsing.readline()


# Prune C3 and only leave 3-tuple items that are greater than support threshold
L3 = {k: v for k, v in C3.items() if v >= s}

# Calculate the confidence of (A,B) -> C and (A,C) -> B and (B,C) -> A from L3 and L2 occurrences
conf_L3_1 = {k: v/L2[(k[0],k[1])] if (k[0],k[1]) in L2.keys() else v/L2[(k[1],k[0])] for k, v in L3.items()}
conf_L3_2 = {(k[1],k[2],k[0]): v/L2[(k[1],k[2])] if (k[1],k[2]) in L2.keys() else v/L2[(k[2],k[1])] for k, v in L3.items()}
conf_L3_3 = {(k[0],k[2],k[1]): v/L2[(k[0],k[2])] if (k[0],k[2]) in L2.keys() else v/L2[(k[0],k[2])] for k, v in L3.items()}

conf_L3 = {**conf_L3_1, **conf_L3_2, **conf_L3_3}

# Sort and get the top 5 rules for 3-tuples
conf_L3_sorted = sorted(conf_L3.items(), key=lambda kv: (-kv[1], kv[0][0][0]))
conf_L3_sorted[0:5]


