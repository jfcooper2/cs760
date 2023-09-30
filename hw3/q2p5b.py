import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import knn
import roc
import logistic

df = pd.read_csv("data/emails.csv", index_col=0)
data = df.values[:,:-1]
classes = df.values[:,-1]

xs_train = data[:1000]
xs_test = data[1000:]

ys_train = classes[:1000]
ys_test = classes[1000:]

ys_model = logistic.logistic(xs_train, xs_test, ys_train)


roc.plotROC(ys_test, ys_model)
print("Test")
print(ys_test)
print("Model")
print(ys_model)
plt.title("Part 5 - ROC for Logistic Regression")
plt.grid()
plt.savefig("img/q2p5b.png")
plt.show()
