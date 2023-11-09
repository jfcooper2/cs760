import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

sigmas = [0.5, 1, 2, 4, 8]

for sigma in sigmas:
    pa_mean = np.array([-1, -1])
    pa_cov = sigma*np.array([[2, 0.5], [0.5, 2]])
    pa = np.random.multivariate_normal(pa_mean, pa_cov, 100)
    
    pb_mean = np.array([1, -1])
    pb_cov = sigma*np.array([[1, -0.5], [-0.5, 2]])
    pb = np.random.multivariate_normal(pb_mean, pb_cov, 100)
    
    pc_mean = np.array([0, 1])
    pc_cov = sigma*np.array([[1, 0], [0, 2]])
    pc = np.random.multivariate_normal(pc_mean, pc_cov, 100)

    """
    plt.scatter(pa[:,0], pa[:,1])
    plt.scatter(pb[:,0], pb[:,1])
    plt.scatter(pc[:,0], pc[:,1])
    """

    xs = np.concatenate((pa, pb, pc))

    # KNN
    best_obj = 1000
    best_acc = 0
    for itr in range(10):
        mus = xs[np.random.choice(np.arange(xs.shape[0]), 3)]
        classes = -1 * np.ones(xs.shape[0])
        while True:
            dists = np.zeros((3, xs.shape[0]))
            dists[0] = np.linalg.norm(mus[0] - xs, axis=1)
            dists[1] = np.linalg.norm(mus[1] - xs, axis=1)
            dists[2] = np.linalg.norm(mus[2] - xs, axis=1)
            classes_new = np.argmin(dists, axis=0)
            if np.all(np.equal(classes,classes_new)):
                break
            classes = classes_new
            try:
                mus[0] = np.mean(xs[classes == 0], axis=0)
                mus[1] = np.mean(xs[classes == 1], axis=0)
                mus[2] = np.mean(xs[classes == 2], axis=0)
            except Exception:
                # Something went wrong with classes
                mus = xs[np.random.choice(np.arange(xs.shape[0]), 3)]

        knn_obj = 0
        knn_obj += np.sum(np.linalg.norm(xs[classes == 0] - mus[0], axis=1))
        knn_obj += np.sum(np.linalg.norm(xs[classes == 1] - mus[1], axis=1))
        knn_obj += np.sum(np.linalg.norm(xs[classes == 2] - mus[2], axis=1))
    
        dists = np.zeros((3,3))
    
        for pi, p in enumerate([pa_mean, pb_mean, pc_mean]):
            for j in range(3):
                dists[pi,j] = np.linalg.norm(mus[j] - p)
    
        class_map = {}
        class_map['a'] = np.argmin(dists[0,:])
        class_map['b'] = np.argmin(dists[1,:])
        if class_map['a'] == class_map['b']:
            #print('Common closest mean')
            class_map['b'] = (class_map['b'] + 1) % 3
        class_map['c'] = 3 - class_map['a'] - class_map['b']
    
        knn_acc = 0
        knn_acc += np.sum(classes[:100] == class_map['a'])
        knn_acc += np.sum(classes[100:200] == class_map['b'])
        knn_acc += np.sum(classes[200:] == class_map['c'])
        knn_acc /= xs.shape[0]

        if best_obj > knn_obj: best_obj = knn_obj
        if best_acc < knn_acc: best_acc = knn_acc

    """
    plt.figure()
    plt.scatter(xs[:,0], xs[:,1], c=classes)
    plt.scatter(mus[:,0], mus[:,1], c='orange')
    plt.title("K-means")
    plt.show()
    """

    # I know this is K-means, but it ruins the clean formating to change it
    print("Sigma: %2f  |  KNN Obj: %.3f  |  KNN Acc: %.3f" % (sigma, best_obj, best_acc))

    # GMMS
    best_obj = 1000
    best_knn_obj = 1000
    best_acc = 0
    for itr in range(20):
        gmm_mus = xs[np.random.choice(np.arange(xs.shape[0]), 3)]
        covs = np.zeros((3,2,2))
        for i in range(3):
            covs[i] = np.eye(2)
        priors = np.ones(3) / 3

        for run in range(500): # 100
            # E
            ws = np.zeros((xs.shape[0], 3))
            for i in range(3):
                ws[:,i] = multivariate_normal.pdf(xs, gmm_mus[i], covs[i])
            ws *= priors
            ws /= np.outer(np.sum(ws, axis=1), np.ones(3))

            # M
            priors = np.sum(ws, axis=0) / xs.shape[0]
            for i in range(3):
                gmm_mus[i] = np.zeros(2)
                for j in range(xs.shape[0]):
                    gmm_mus[i] += ws[j,i] * xs[j]
                gmm_mus[i] /= np.sum(ws[:,i])

                covs[i] = np.zeros((2,2))
                for j in range(xs.shape[0]):
                    covs[i] += ws[j,i] * np.outer(xs[j] - gmm_mus[i], xs[j] - gmm_mus[i])
                covs[i] /= np.sum(ws[:,i])

        classes = np.argmax(ws, axis=1)

        knn_obj = 0
        knn_obj += np.sum(np.linalg.norm(xs[classes == 0] - mus[0], axis=1))
        knn_obj += np.sum(np.linalg.norm(xs[classes == 1] - mus[1], axis=1))
        knn_obj += np.sum(np.linalg.norm(xs[classes == 2] - mus[2], axis=1))

        gmm_obj = 0
        for i in range(xs.shape[0]):
            gmm_obj += -np.log(ws[i, classes[i]])
    
        dists = np.zeros((3,3))
    
        for pi, p in enumerate([pa_mean, pb_mean, pc_mean]):
            for j in range(3):
                dists[pi,j] = np.linalg.norm(gmm_mus[j] - p)
    
        class_map = {}
        class_map['a'] = np.argmin(dists[0,:])
        class_map['b'] = np.argmin(dists[1,:])
        if class_map['a'] == class_map['b']:
            #print('Common closest mean')
            class_map['b'] = (class_map['b'] + 1) % 3
        class_map['c'] = 3 - class_map['a'] - class_map['b']
    
        gmm_acc = 0
        gmm_acc += np.sum(classes[:100] == class_map['a'])
        gmm_acc += np.sum(classes[100:200] == class_map['b'])
        gmm_acc += np.sum(classes[200:] == class_map['c'])
        gmm_acc /= xs.shape[0]

        if best_obj > gmm_obj: best_obj = gmm_obj
        if best_knn_obj > knn_obj: best_knn_obj = knn_obj
        if best_acc < gmm_acc: best_acc = gmm_acc

    """
    plt.figure()
    plt.scatter(xs[:,0], xs[:,1], c=classes)
    plt.scatter(gmm_mus[:,0], gmm_mus[:,1], c='orange')
    plt.title("GMM")
    plt.show()
    """

    print("Sigma: %2f  |  GMM Obj: %.3f  |  GMM Acc: %.3f" % (sigma, best_knn_obj, best_acc))
    #print("Sigma: %2f  |  GMM Obj: %.3f  |  GMM Acc: %.3f" % (sigma, knn_obj, best_acc))
