import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

np.random.seed(43)

a,b = 0,6.28
n = 10
eps_list = [10 ** i for i in range(-5,0)]

samples = np.random.uniform(low=a, high=b, size=n)
xs_train = samples
ys_train = np.sin(xs_train)

poly = lagrange(xs_train, ys_train)

samples = np.random.uniform(low=a, high=b, size=n)
xs_test = samples
ys_test = np.sin(xs_test)
ys_model = poly(xs_test)

print("n: %d" % n)
print()
print("Train error L2:", np.linalg.norm(poly(xs_train)-ys_train, ord=2))
print()
print("Test error L2:", np.linalg.norm(ys_model-ys_test, ord=2))
print()

errors = []

for eps in eps_list:
    samples = np.random.uniform(low=a, high=b, size=n)
    xs_train = samples
    ys_train = np.sin(xs_train)
    ys_test = np.sin(xs_train) + eps * np.random.normal(size=n)
    
    poly = lagrange(xs_train, ys_train)
    
    samples = np.random.uniform(low=a, high=b, size=n)
    xs_test = samples
    ys_test = np.sin(xs_test) + eps * np.random.normal(size=n)
    ys_model = poly(xs_test)

    errors.append(np.linalg.norm(ys_model-ys_test, ord=2))

    print("eps:", eps)
    print("Test error L2:", errors[-1])
    print()

plt.loglog(eps_list, errors)
plt.title("Errors for Varying Amounts of Noise")
plt.xlabel("log(eps)")
plt.ylabel("log(Error) ($L^2$)")
plt.savefig("img/p4.png")
plt.show()
