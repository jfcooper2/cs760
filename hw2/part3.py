import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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

    tree = DecisionTreeClassifier()
    tree.fit(xs_train, ys_train)

    sizes.append(2*tree.get_n_leaves()-1) # Number of internal nodes and leaves
    ys_model = tree.predict(xs_test)
    errors.append(1-np.sum(ys_model == ys_test)/len(ys_test))
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
plt.savefig("img/p3lc.png")

plt.show()
