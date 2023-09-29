import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import knn

df = pd.read_csv("data/emails.csv", index_col=0)
data = df.values[:,:-1]
classes = df.values[:,-1]

ks = [1, 3, 5, 7, 10]
all_accuracies = []
for k in [1, 3, 5, 7, 10]:
    accuracies, precisions, recalls = knn.knn_crossval(data, classes, k=k)
    all_accuracies.append(np.mean(accuracies))

print("Accuracies")
print(all_accuracies)

plt.plot(ks, all_accuracies)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.title("Part 3 - Accurcies For Different ks")
plt.grid()
plt.savefig("img/q2p4.png")
plt.show()
