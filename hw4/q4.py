import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def sigma(x):
    return 1 / (1 + np.exp(-x))

def g(x):
    return (1 / np.sum(np.exp(x))) * np.exp(x)

# Build the models
d = 28*28
d1 = 40
k = 10

W1 = np.zeros((d1, d))
W2 = np.zeros((k, d1))

#W1 = np.random.random(size=(d1, d)) * 0.1 - 0.05
#W2 = np.random.random(size=(k, d1)) * 0.1 - 0.05

# Model parameters
eta = 5e-5

# Get data
train_data = MNIST(root="MNIST/", train=True, download=True)
test_data = MNIST(root="MNIST/", train=False, download=True)
#train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

itrs = []
train_errs = []
test_errs = []

test_xs = []
test_labels = []
for img, label in test_data:
    test_xs.append(list(img.getdata()))
    test_labels.append(label)

test_xs = np.array(test_xs) / np.max(test_xs)
test_ys = np.zeros((len(test_labels), 10)) 
for i,label in enumerate(test_labels):
    test_ys[i][label] = 1 # One hot encodings

xs = []
labels = []
for img, label in train_data:
    xs.append(list(img.getdata()))
    #print(len(xs[-1]))
    labels.append(label)

xs = np.array(xs) / np.max(xs)
ys = np.zeros((len(labels), 10)) 
for i,label in enumerate(labels):
    ys[i][label] = 1 # One hot encodings

for itr in range(xs.shape[0]):
    if itr % 10 == 0:
        print("Itr", itr)
    #points = np.random.choice(range(xs.shape[0]), size=1) # Minibatch Size
    points = [itr]
    #points = range(xs.shape[0])
    #points = [0]

    dLdW1_full = np.zeros_like(W1)
    dLdW2_full = np.zeros_like(W2)
    for index, point in enumerate(points):
        # Pick data points    
        x = xs[point]
        y = ys[point]
        
        # Forward pass
        z1 = np.dot(W1, x)
        a1 = sigma(z1)
        z2 = np.dot(W2, a1)
        yhat = g(z2)
        if index == 0 and False:
            #print("z2", z2)
            #print("y", y)
            #print("W1", np.sum(W1, axis=1))
            print(a1)

        # Backward pass
        dLdW2 = np.outer(yhat - y, a1)
        dLdW1 = np.outer(np.multiply(np.dot(W2.T, yhat - y), np.multiply(a1, 1 - a1)), x)

        #if index == 0:
        #    print(a1)

        dLdW1_full += dLdW1
        dLdW2_full += dLdW2

    # Update the weights
    W1 -= eta * (1/len(points)) * dLdW1_full
    W2 -= eta * (1/len(points)) * dLdW2_full

    loss = 0
    for point in range(xs.shape[0] // 500):
        x = xs[point]
        y = ys[point]
        yhat = g(np.dot(W2, sigma(np.dot(W1, x))))
        #if np.argmax(yhat) != np.argmax(y):
        #    loss += 1
        loss += -np.log(np.sum(np.multiply(y, yhat)))
    loss /= (xs.shape[0] // 500)
    train_errs.append(loss)

    loss = 0
    for point in range(test_xs.shape[0] // 100):
        x = test_xs[point]
        y = test_ys[point]
        yhat = g(np.dot(W2, sigma(np.dot(W1, x))))
        #if np.argmax(yhat) != np.argmax(y):
        #    loss += 1
        loss += -np.log(np.sum(np.multiply(y, yhat)))
    loss /= (test_xs.shape[0] // 100)
    test_errs.append(loss)

    itrs.append(itr)

for index in range(0,xs.shape[0],10):
    pass
    #x = xs[index]
    print(np.argmax(ys[index]) - np.argmax(g(np.dot(W2, sigma(np.dot(W1, x))))))
    #print((g(np.dot(W2, sigma(np.dot(W1, x))))))
    #print(np.argmax(g(np.dot(W2, sigma(np.dot(W1, x))))))

plt.plot(itrs, train_errs)
plt.plot(itrs, test_errs)
plt.legend(["Train", "Test"])
plt.title("Learning Curve")
plt.xlabel("Itr")
plt.ylabel("Loss")
plt.grid('minor')
plt.savefig("img/p4_learncurve.png")
plt.show()
