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


def knn(xs_train, xs_test, classes_train, k=1):
    n = xs_train.shape[0]
    classes_test = []
    for row in range(xs_test.shape[0]):
        dists = []
        for i in range(n):
            dists.append(np.linalg.norm(xs_train[i] - xs_test[row], ord=2))
    
        idx = list(range(n))
        idx.sort(key = lambda i: dists[i])
    
        closest_k = idx[:k]
        closest_k_classes = classes_train[closest_k]
        ones  = np.sum(closest_k_classes == 1)
        zeros = np.sum(closest_k_classes == 0)
        if ones > zeros:
            classes_test.append(1)
        else:
            classes_test.append(0)
    return np.array(classes_test)

def knn_crossval(xs, classes, k=1, fold=5):
    n_all = xs.shape[0] * 1.0
    accuracies = []
    precisions = []
    recalls    = []
    for f in range(fold):
        print(f)
        train_idx = list(range(int(f*n_all/fold), int((f+1)*n_all/fold)))
        test_idx = list(set(range(int(n_all))).difference(set(train_idx)))

        xs_train = xs[train_idx]
        xs_test  = xs[test_idx]
        classes_train = classes[train_idx]
        classes_true  = classes[test_idx]

        n = xs_train.shape[0]
        classes_test = []
        for row in range(xs_test.shape[0]):
            if row % 100 == 0:
                print("-", row)
            dists = []
            for i in range(n):
                dists.append(np.linalg.norm(xs_train[i] - xs_test[row], ord=2))
        
            idx = list(range(n))
            idx.sort(key = lambda i: dists[i])
        
            closest_k = idx[:k]
            closest_k_classes = classes_train[closest_k]
            ones  = np.sum(closest_k_classes == 1)
            zeros = np.sum(closest_k_classes == 0)
            if ones > zeros:
                classes_test.append(1)
            else:
                classes_test.append(0)
        classes_test = np.array(classes_test)

        accuracies.append(accuracy(classes_true, classes_test))
        precisions.append(precision(classes_true, classes_test))
        recalls.append(recall(classes_true, classes_test))
    return accuracies, precisions, recalls
