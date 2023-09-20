import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from trees import *

def loaddata(datafilename):
    xs, ys = [], []
    with open(datafilename) as infile:
        data = infile.read()
    for line in data.split('\n'):
        if len(line.split()) != 3:
            continue
        xnew1, xnew2, ynew = line.split()
        xnew1 = float(xnew1)
        xnew2 = float(xnew2)
        ynew  = int(ynew)
        xs.append([xnew1, xnew2])
        ys.append(ynew)

    return xs, ys

print("--- Q2 ---")
xs = [[1,1],[1,-1],[-1,1],[-1,-1]]
ys = [1,0,0,1]
q2tree = DecisionTree()
q2tree.fit(xs, ys)
print("Size:", q2tree.head.size())
print("(size is the number of nodes in the tree)")
plt.figure()
plt.scatter([1,-1], [1,-1], s=8, c='r')
plt.scatter([1,-1], [-1,1], s=8, c='b')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.title("Question 2 - Deadlock Data")
plt.savefig("img/q2.png")

print()
print("--- Q3 ---")
xs, ys = loaddata("data/Druns.txt")

mytree = DecisionTree()
mytree.fit(xs, ys)
print("Accuracy:", mytree.score(xs, ys))
mytree.head.show()

print()
print("--- Q4 ---")
xs, ys = loaddata("data/D3leaves.txt")

mytree = DecisionTree()
mytree.fit(xs, ys)
print("Accuracy:", mytree.score(xs, ys))
mytree.show()
plt.figure()
mytree.visualize(xs, ys)

print()
print("--- Q5 ---")
print(" - D1 - ")
xs, ys = loaddata("data/D1.txt")

mytree = DecisionTree()
mytree.fit(xs, ys)
print("Accuracy:", mytree.score(xs, ys))
mytree.show()

print()
print(" - D2 - ")
xs, ys = loaddata("data/D2.txt")

mytree = DecisionTree()
mytree.fit(xs, ys)
print("Accuracy:", mytree.score(xs, ys))
mytree.show()

print()
print("--- Q6 ---")
xs, ys = loaddata("data/D1.txt")

mytree = DecisionTree()
mytree.fit(xs, ys)
plt.figure()
plt.title("D1 Decision Regions")
mytree.visualize(xs, ys)
plt.savefig("img/q6d1.png")

xs, ys = loaddata("data/D2.txt")

mytree = DecisionTree()
mytree.fit(xs, ys)
plt.figure()
plt.title("D2 Decision Regions")
mytree.visualize(xs, ys)
plt.savefig("img/q6d2.png")

print()
print("--- Q7 ---")
xs, ys = loaddata("data/Dbig.txt")

idx = np.array(list(range(len(ys))))
np.random.seed(42)
np.random.shuffle(idx)

xs = np.array(xs)
ys = np.array(ys)
idx = np.array(idx)

xs = xs[idx]
ys = ys[idx]

ns = [32, 128, 512, 2048, 8192]
sizes = []
errors = []

for n in ns:
    xs_train = xs[:n]
    ys_train = ys[:n]
    xs_test = xs[n:]
    ys_test = ys[n:]

    tree = DecisionTree()
    tree.fit(xs_train, ys_train)

    plt.figure()
    plt.title("$D_{%d}$ Decision Regions" % n)
    tree.visualize(xs, ys, doscatter = False)
    plt.savefig("img/q7d%d.png" % n)

    sizes.append(tree.head.size())
    errors.append(1-tree.score(xs_test, ys_test))
    print("n:", n)
    print("error:", errors[-1])
    print("nodes:", sizes[-1])
    print()

plt.figure()
plt.plot(ns, errors)
plt.grid('minor')
plt.title("Learning curve")
plt.xlabel("n")
plt.ylabel("Error")
plt.savefig("img/q7lc.png")








plt.show()
