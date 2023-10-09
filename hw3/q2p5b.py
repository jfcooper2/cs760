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

ys_model_knn = knn.knn(xs_train, xs_test, ys_train, k=5)

aoc_knn = roc.plotROC(ys_test, ys_model_knn)
aoc_log = roc.plotROC(ys_test, ys_model)
plt.legend(["KNN - %.3f" % aoc_knn, "Logistic - %.3f" % aoc_log])
print("Test")
print(ys_test)
print("Model")
print(ys_model)
plt.title("Part 5 - ROC")
plt.grid()
plt.savefig("img/q2p5b.png")
plt.show()
