import numpy as np
import matplotlib.pyplot as plt
import roc
from sklearn import metrics

ys = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
thress = [0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 0.8, 0.85, 0.95]

ys = np.array(ys)
thress = np.array(thress)

roc.plotROC(ys, thress)
plt.savefig("img/q1p5.png")
plt.plot([0, 0.5], [0.5, 1], linewidth=1)
plt.show()
