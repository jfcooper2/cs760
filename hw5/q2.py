import numpy as np
import matplotlib.pyplot as plt

def buggyPCA(data, d):
    u, s, vt = np.linalg.svd(data)
    vt = vt[:d]
    z = vt @ data.T
    if len(z.shape) == 1:
        z = np.expand_dims(z, axis=-1)
    x = (vt.T @ z).T
    return z, x

def demeanedPCA(data, d):
    mu = np.mean(data, axis=0)
    data_ = (data - mu)
    u, s, vt = np.linalg.svd(data_)
    vt = vt[:d]
    z = vt @ data_.T
    if len(z.shape) == 1:
        z = np.expand_dims(z, axis=-1)
    x = (vt.T @ z).T + mu
    return z, x

def normalizedPCA(data, d):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_ = ((data - mu) / std)
    u, s, vt = np.linalg.svd(data_)
    vt = vt[:d]
    z = vt @ data_.T
    if len(z.shape) == 1:
        z = np.expand_dims(z, axis=-1)
    x = ((vt.T @ z).T * std) + mu
    return z, x

def dro(data, d):
    u, s, vt = np.linalg.svd(data)
    s = np.diag(s)
    up = u[:,:(d+1)]
    sp = s[:(d+1),:(d+1)]
    vtp = vt[:(d+1),:]
    x = up @ sp @ vtp
    b = np.mean(x, axis=0) # Find the affine part
    x -= b
    u, s, vt = np.linalg.svd(x)
    print(s)
    s = np.diag(s)
    up = u[:,:d]
    sp = s[:d,:d]
    vtp = vt[:d,:]

    zp = up
    zp_bar = np.mean(zp, axis=0)
    ap = (sp @ vtp).T
    #ap = (1/data.shape[0]) * (sp @ vtp).T

    m = np.cov(zp.T)
    if d == 1:
        m = np.array([[m]])
    mu, ms, mvt = np.linalg.svd(m)
    ms = (1/(data.shape[0]-1)) * np.diag(np.divide(np.ones(d), ms))
    m = np.sqrt(ms) @ mvt

    z = (zp - zp_bar) @ m.T
    a = ap @ np.linalg.inv(m)
    b = b + np.dot(a, np.dot(m.T, zp_bar))

    x = (a @ z.T).T + b
    return z, x



############################################

print()
print("D2")
print()

with open("data/data2D.csv") as infile:
    lines = infile.readlines()

data2D = []
for line in lines:
    data2D.append(line.split(','))
data2D = np.array(data2D, dtype=np.float32)

plt.figure()
z, x = buggyPCA(data2D, 1)
plt.scatter(data2D[:,0], data2D[:,1])
plt.scatter(x[:,0], x[:,1], marker='x')
plt.legend(["True", "Recovered"])
plt.title("Buggy PCA")
plt.savefig("img/buggy2.png")
print("Buggy Error:", np.sum(np.square(data2D - x)))


data2D = []
for line in lines:
    data2D.append(line.split(','))
data2D = np.array(data2D, dtype=np.float32)

plt.figure()
z, x = demeanedPCA(data2D, 1)
plt.scatter(data2D[:,0], data2D[:,1])
plt.scatter(x[:,0], x[:,1], marker='x')
plt.legend(["True", "Recovered"])
plt.title("Demeaned PCA")
plt.savefig("img/demeaned2.png")
print("Demeaned Error:", np.sum(np.square(data2D - x)))


data2D = []
for line in lines:
    data2D.append(line.split(','))
data2D = np.array(data2D, dtype=np.float32)

plt.figure()
z, x = normalizedPCA(data2D, 1)
plt.scatter(data2D[:,0], data2D[:,1])
plt.scatter(x[:,0], x[:,1], marker='x')
plt.legend(["True", "Recovered"])
plt.title("Normalized PCA")
plt.savefig("img/normalized2.png")
print("Normalized Error:", np.sum(np.square(data2D - x)))


data2D = []
for line in lines:
    data2D.append(line.split(','))
data2D = np.array(data2D, dtype=np.float32)

plt.figure()
z, x = dro(data2D, 1)
plt.scatter(data2D[:,0], data2D[:,1])
plt.scatter(x[:,0], x[:,1], marker='x')
plt.title("DRO")
plt.savefig("img/dro.png")
print("DRO Error:", np.sum(np.square(data2D - x)))

############################################

print()
print("D1000")
print()

arr = []
with open("data/data1000D.csv") as infile:
    lines = infile.readlines()

data1000D = []
for line in lines:
    data1000D.append(line.split(','))
data1000D = np.array(data1000D, dtype=np.float32)


u, s, vt = np.linalg.svd(data1000D)
plt.figure()
plt.loglog(s)
plt.title("Singular values for D1000 Data")
plt.xlabel("n")
plt.ylabel("$\sigma$")
plt.grid('minor')
plt.savefig("img/singular.png")


z, x = buggyPCA(data1000D, 12)
print("Buggy Error:", np.sum(np.square(data1000D - x)))

z, x = demeanedPCA(data1000D, 12)
print("Demeaned Error:", np.sum(np.square(data1000D - x)))

z, x = normalizedPCA(data1000D, 12)
print("Normalized Error:", np.sum(np.square(data1000D - x)))

z, x = dro(data1000D, 12)
print("DRO Error:", np.sum(np.square(data1000D - x)))


plt.show()
