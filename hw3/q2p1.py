import numpy as np
import matplotlib.pyplot as plt
import knn

with open("data/D2z.txt") as infile:
    lines = infile.readlines()

train_xs = []
train_classes = []
for line in lines:
    x, y, c = line.split()
    train_xs.append([float(x), float(y)])
    train_classes.append(int(c))

n = len(train_xs)
train_xs = np.array(train_xs)
train_classes = np.array(train_classes)

test_xs = []
for i in np.linspace(-2, 2, 21):
    for j in np.linspace(-2, 2, 21):
        test_xs.append([i,j])
test_xs = np.array(test_xs)

test_classes = knn.knn(train_xs, test_xs, train_classes)

plt.scatter(train_xs[:,0], train_xs[:,1], c=train_classes, marker='.')
plt.scatter(test_xs[:,0], test_xs[:,1], c=test_classes)
plt.title("Part 1 1-NN Plot")
plt.savefig("img/q2p1.png")
plt.show()
