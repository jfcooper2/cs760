import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import knn

df = pd.read_csv("data/emails.csv", index_col=0)
data = df.values[:,:-1]
classes = df.values[:,-1]

accuracies, precisions, recalls = knn.knn_crossval(data, classes)

print("Accuracies")
print(accuracies)
print("Precisions")
print(precisions)
print("Recalls")
print(recalls)
