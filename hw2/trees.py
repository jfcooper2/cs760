import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, value):
        self.value = value
        self.is_leaf = True

    def set_children(self, thres, dim):
        # Convert a node to a true node instead of a leaf
        self.thres = thres
        self.dim = dim

        idx = self.xs[:,dim].argsort()
        self.xs = self.xs[idx]
        self.ys = self.ys[idx]

        index = np.min(np.argwhere(self.xs[:,dim]>thres))
        leftxs = self.xs[:index]
        rightxs = self.xs[index:]
        leftys = self.ys[:index]
        rightys = self.ys[index:]

        if np.sum(leftys) >= len(leftys)/2:
            self.left = Node(1)
        else:
            self.left = Node(0)
        self.left.xs = leftxs
        self.left.ys = leftys

        if np.sum(rightys) >= len(rightys)/2:
            self.right = Node(1)
        else:
            self.right = Node(0)
        self.right.xs = rightxs
        self.right.ys = rightys

        self.is_leaf = False
        
    def eval(self, x):
        if self.is_leaf:
            return self.value
        else:
            if x[self.dim] <= self.thres:
                return self.left.eval(x)
            else:
                return self.right.eval(x)

    def size(self):
        if self.is_leaf:
            return 1
        return 1 + self.right.size() + self.left.size()

    def show(self, depth=0):
        if self.is_leaf:
            print(depth * "  " + str(self.value))
        else:
            print(depth * "  " + "x_" + str(self.dim+1) + "<=" + str(self.thres))
            self.left.show(depth+1)
            print(depth * "  " + "x_" + str(self.dim+1) + ">" + str(self.thres))
            self.right.show(depth+1)

    def visualize(self, xlims, ylims):
        if self.is_leaf:
            if self.value == 0:
                plt.fill_between(xlims, ylims[0], ylims[1], facecolor='r', alpha=0.2)
            else:
                plt.fill_between(xlims, ylims[0], ylims[1], facecolor='b', alpha=0.2)
        else:
            if self.dim == 0:
                self.left.visualize([xlims[0], self.thres], ylims)
                self.right.visualize([self.thres, xlims[1]], ylims)

            if self.dim == 1:
                self.left.visualize(xlims, [ylims[0], self.thres])
                self.right.visualize(xlims, [self.thres, ylims[1]])



class DecisionTree:
    def __init__(self):
        self.head = None

    def entropy(self, ys):
        zeros = np.sum(ys == 0) / len(ys)
        ones = np.sum(ys == 1) / len(ys)
        if zeros > 0 and ones > 0:
            return - (zeros * np.log(zeros) / np.log(2)) - (ones * np.log(ones) / np.log(2))
        if zeros > 0 and ones == 0:
            return - (zeros * np.log(zeros) / np.log(2)) 
        if zeros == 0 and ones > 0:
            return - (ones * np.log(ones) / np.log(2))
        if zeros == 0 and ones == 0:
            return 0
    
    def fit(self, xs, ys):
        xs = np.array(xs)
        ys = np.array(ys)

        assert xs.shape[1] == 2 # 2 dimensional input vectors
        n = len(ys)

        # Make the most basic tree where it does a majority vote
        if np.sum(ys) >= n/2:
            self.head = Node(1)
        else:
            self.head = Node(0)
        # Load the data for that node in the node (for finding splits)
        self.head.xs = np.array(xs)
        self.head.ys = np.array(ys)

        currleaves = [self.head]

        # Work through the leaves, finding splits for them as needed
        while len(currleaves) > 0:
            currleaf = currleaves[0]
            currleaves = currleaves[1:]

            if len(currleaf.ys) == 0:
                continue

            best_split = None
            best_gain = -np.float('inf')
            for d in range(2): # the dimension
                idx = currleaf.xs[:,d].argsort()
                currleaf.xs = currleaf.xs[idx]
                currleaf.ys = currleaf.ys[idx]
                cidx = (np.argwhere(currleaf.ys[:-1] != currleaf.ys[1:])).ravel()
                threslist = currleaf.xs[cidx,d]
                for thres in threslist:
                    leftidx = currleaf.xs[:,d] <= thres
                    leftys = currleaf.ys[leftidx]
                    leftp = len(leftys)/len(currleaf.ys)
                    
                    rightidx = currleaf.xs[:,d] > thres
                    rightys = currleaf.ys[rightidx]
                    rightp = len(rightys)/len(currleaf.ys)

                    # Issues happen when sorting points with identical coordinates
                    if leftp == 0 or rightp == 0:
                        continue

                    mutual = self.entropy(currleaf.ys)
                    mutual -= (len(leftys)/len(currleaf.ys)) * self.entropy(leftys)
                    mutual -= (len(rightys)/len(currleaf.ys)) * self.entropy(rightys)

                    split_entropy  = -(leftp  * np.log(leftp)  / np.log(2))
                    split_entropy += -(rightp * np.log(rightp) / np.log(2))

                    gain = mutual / split_entropy
                    if type(best_split) == type(None) or gain > best_gain:
                        best_gain = gain
                        best_split = (thres,d)

            if best_gain <= 0:
                continue

            currleaf.set_children(*best_split)
            currleaves.append(currleaf.left)
            currleaves.append(currleaf.right)


    def eval(self, xs):
        ys = []
        for row in range(len(xs)):
            ys.append(self.head.eval(xs[row]))
        return np.array(ys)

    def score(self, xs, ys, error="accuracy"):
        ys_model = self.eval(xs)
        ys = np.array(ys)
        ys_model = np.array(ys_model)

        if error == "accuracy":
            return np.sum(ys == ys_model)/(len(ys))

    def show(self):
        self.head.show()

    def visualize(self, xs, ys, doscatter = True):
        xs = np.array(xs)
        ys = np.array(ys)
        xlims = [np.min(xs[:,0])-1, np.max(xs[:,0])+1]
        ylims = [np.min(xs[:,1])-1, np.max(xs[:,1])+1]

        zerosidx = np.argwhere(ys == 0)
        onesidx = np.argwhere(ys == 1)

        if doscatter:
            plt.scatter(xs[zerosidx,0], xs[zerosidx,1], c='r', s=5)
            plt.scatter(xs[onesidx,0], xs[onesidx,1], c='b', s=5)

        self.head.visualize(xlims, ylims)





