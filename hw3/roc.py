import numpy as np
import matplotlib.pyplot as plt

def findROC(ys, thress):
    idx = np.argsort(thress)[::-1]
    ys = ys[idx]
    thress = thress[idx]

    tprs = []
    fprs = []

    cs = [1]
    cs.extend(thress)
    cs.append(0)

    for c in cs:
        yhats = np.array(thress) > c
        tp = np.sum(np.multiply(ys == 1, yhats == 1))
        fp = np.sum(np.multiply(ys == 0, yhats == 1))
        t = np.sum(ys == 1)
        f = np.sum(ys == 0)
        
        #print(c, ":", tp, fp, tn, fn)

        tpr = tp / t
        fpr = fp / f

        print(tpr, fpr)

        if len(tprs) != 0:
            tprs.append(tpr)
            fprs.append(fprs[-1])
        tprs.append(tpr)
        fprs.append(fpr)

    tprs.append(1)
    fprs.append(fprs[-1])
    tprs.append(1)
    fprs.append(1)

    return tprs, fprs

def plotROC(ys, thress, show=False):
    tprs, fprs = findROC(ys, thress)

    plt.plot(fprs, tprs)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")

    plt.plot([0,1], [0,1])

    if show:
        plt.show()
