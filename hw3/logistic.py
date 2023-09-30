import numpy as np
import matplotlib.pyplot as plt

def accuracy(ys_true, ys_model):
    tp = np.sum(np.multiply(ys_true == 1, ys_model == 1))
    tn = np.sum(np.multiply(ys_true == 0, ys_model == 0))
    p  = np.sum(ys_true == 1)
    n  = np.sum(ys_true == 0)
    return (tp + tn) / (p + n)

def precision(ys_true, ys_model):
    tp = np.sum(np.multiply(ys_true == 1, ys_model == 1))
    fp = np.sum(np.multiply(ys_true == 0, ys_model == 1))
    return tp / (tp + fp)

def recall(ys_true, ys_model):
    tp = np.sum(np.multiply(ys_true == 1, ys_model == 1))
    fn = np.sum(np.multiply(ys_true == 1, ys_model == 0))
    return tp / (tp + fn)

def sigmoid(xs):
    return np.divide(1, 1 + np.exp(xs))


def logistic(xs_train, xs_test, classes_train, eta=1e-4):
    n = xs_train.shape[0]
    d = xs_train.shape[1]
    theta = np.zeros(d)
    for itr in range(100):
        f = sigmoid(np.dot(xs_train, theta)) # indexed over all i in n
        #print(np.sum(f>0.5))
        dtheta = np.zeros(d)
        for i in range(n):
            dtheta += (1.0/n) * (f[i] - classes_train[i]) * xs_train[i]
        if itr % 10 == 0:
            print(np.max(np.abs(dtheta)))
        theta += eta * dtheta
    print("Theta:", theta)

    f = sigmoid(np.dot(xs_test, theta))
    print(np.sum(f > 0.5))
    return f # sigmoid related output

def logistic_crossval(xs, classes, k=1, fold=5):
    xs = np.insert(xs, -1, 1, axis=1)
    n_all = xs.shape[0] * 1.0
    accuracies = []
    precisions = []
    recalls    = []
    for f in range(fold):
        print("Fold:", f)
        train_idx = list(range(int(f*n_all/fold), int((f+1)*n_all/fold)))
        test_idx = list(set(range(int(n_all))).difference(set(train_idx)))

        xs_train = xs[train_idx]
        xs_test  = xs[test_idx]
        classes_train = classes[train_idx]
        classes_true  = classes[test_idx]

        classes_test = logistic(xs_train, xs_test, classes_train) > 0.5

        accuracies.append(accuracy(classes_true, classes_test))
        precisions.append(precision(classes_true, classes_test))
        recalls.append(recall(classes_true, classes_test))
    return accuracies, precisions, recalls
