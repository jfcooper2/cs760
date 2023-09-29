import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

np.random.seed(43)

a,b = 0,100
#n = 100
n = 100
eps_list = [3 ** i for i in range(-14,10)]

#samples = np.random.uniform(low=a, high=b, size=n)
samples = np.linspace(a, b, n)
xs_train = samples
ys_train = np.sin(xs_train)

poly = lagrange(xs_train, ys_train)

#samples = np.random.uniform(low=a, high=b, size=n)
#samples = np.linspace(a, b, n)
xs_test = samples
ys_test = np.sin(xs_test)
ys_model = poly(xs_test)

plt.scatter(xs_test, ys_model)
#plt.yscale('log')
plt.title('Recovered Train Set Poly')
plt.savefig('img/p4_poly.png')
plt.figure()

print("n: %d" % n)
print()
print("Train error L2:", np.linalg.norm(poly(xs_train)-ys_train, ord=2))
print()
print("Test error L2:", np.linalg.norm(ys_model-ys_test, ord=2))
print()

test_errors = []
train_errors = []

for eps in eps_list:
    xs_train = samples + eps * np.random.normal(size=n)
    ys_train = np.sin(xs_train)
    poly = lagrange(xs_train, ys_train)

    xs_test = samples
    ys_test = np.sin(xs_test)
    ys_model = poly(xs_test)
    #plt.scatter(xs_test, ys_model)
    #plt.show()

    test_errors.append(np.linalg.norm(ys_model-ys_test, ord=2))
    train_errors.append(np.linalg.norm(poly(xs_train)-ys_train, ord=2))

    print("eps:", eps)
    print("Test error L2:", test_errors[-1])
    print("Train error L2:", train_errors[-1])
    print()


plt.loglog(eps_list, test_errors)
plt.loglog(eps_list, train_errors)
plt.legend(["test", "train"])
plt.title("Errors for Varying Amounts of Noise")
plt.xlabel("log(eps)")
plt.ylabel("log(Error) ($L^2$)")
plt.savefig("img/p4.png")
plt.show()
