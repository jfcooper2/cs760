import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Get data
train_data = MNIST(root="MNIST/", train=True, download=True)
test_data = MNIST(root="MNIST/", train=False, download=True)

xs = []
labels = []
for img, label in train_data:
    xs.append(list(img.getdata()))
    #print(len(xs[-1]))
    labels.append(label)

xs = np.array(xs)
ys = np.zeros((len(labels), 10)) 
for i,label in enumerate(labels):
    ys[i][label] = 1 # One hot encodings
#ys = np.array(labels)

xs = torch.tensor(xs, dtype=torch.float32)
ys = torch.tensor(ys, dtype=torch.float32)

xs_test = []
labels_test = []
for img, label in test_data:
    xs_test.append(list(img.getdata()))
    #print(len(xs[-1]))
    labels_test.append(label)

xs_test = np.array(xs_test)
ys_test = np.zeros((len(labels_test), 10)) 
for i,label in enumerate(labels_test):
    ys_test[i][label] = 1 # One hot encodings
#ys = np.array(labels)

xs_test = torch.tensor(xs_test, dtype=torch.float32)
ys_test = torch.tensor(ys_test, dtype=torch.float32)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.W1 = nn.Linear(28*28, 40, bias=False)  # 5*5 from image dimension
        self.W2 = nn.Linear(40, 10, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.sigmoid(self.W1(x))
        x = self.sm(self.W2(x))
        return x


model = Net()
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.01)

n_epochs = 500
batch_size = 100

train_losses = []
test_losses = []
epochs = []

for epoch in range(n_epochs):
    for i in range(0, len(xs), batch_size):
        Xbatch = xs[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = ys[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(loss_fn(model(xs), ys).item())
    test_losses.append(loss_fn(model(xs_test), ys_test).item())
    epochs.append(epoch)
    print(f'Finished epoch {epoch}, latest loss {loss}')

yhats_test = model(xs_test)
for index in range(xs_test.shape[0]):
    print(torch.argmax(ys_test[index]).item() - torch.argmax(yhats_test[index]).item())

plt.plot(epochs, train_losses)
plt.plot(epochs, test_losses)
plt.legend(["Train Errors", "Test Errors"])
plt.xlabel("epoch")
plt.ylabel("error")
plt.title("Learning Curve")
plt.grid('minor')
plt.savefig("img/q4pt.png")
plt.show()
