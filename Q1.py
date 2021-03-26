import numpy as np
import matplotlib.pyplot as plt


def Availability(n, n2, dim, stride):
    if dim == 1:
        if n < 0 or n >= stride:
            return False
    else:
        if n < 0 or n >= stride or n2 < 0 or n2 >= stride:
            return False
        else:
            return True


def Update(R, Rect, k, change, Alpha, data, weight, winner, stride, num_n):
    change[k] = np.max(abs(Alpha * (data[k] - weight[
        winner])))  # inja miad ekhtelafe tak tak pixel haye y tasvir ro hesab mikone max migire
    weight[winner] = weight[winner] - Alpha * (data[k] - weight[winner])
    if R == 1 and Rect == 0:
        if Availability(winner - 1, 0, 1, num_n) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[winner - 1]))))
            weight[winner - 1] = weight[winner - 1] - Alpha * (data[k] - weight[winner - 1])
        if Availability(winner + 1, 0, 1, num_n) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[winner + 1]))))
            weight[winner + 1] = weight[winner + 1] - Alpha * (data[k] - weight[winner + 1])
    if Rect == 1:
        nei1 = int(winner / stride)
        nei2 = (winner % stride) - 1
        if Availability(nei1, nei2, 2, stride) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[nei1 * stride + nei2]))))
            weight[nei1 * stride + nei2] = weight[nei1 * stride + nei2] - Alpha * (
                    data[k] - weight[nei1 * stride + nei2])

        nei1 = int(winner / stride)
        nei2 = (winner % stride) + 1
        if Availability(nei1, nei2, 2, stride) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[nei1 * stride + nei2]))))
            weight[nei1 * stride + nei2] = weight[nei1 * stride + nei2] - Alpha * (
                    data[k] - weight[nei1 * stride + nei2])

        nei1 = int(winner / stride) - 1
        nei2 = (winner % stride) - 1
        if Availability(nei1, nei2, 2, stride) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[nei1 * stride + nei2]))))
            weight[nei1 * stride + nei2] = weight[nei1 * stride + nei2] - Alpha * (
                    data[k] - weight[nei1 * stride + nei2])

        nei1 = int(winner / stride) - 1
        nei2 = (winner % stride)
        if Availability(nei1, nei2, 2, stride) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[nei1 * stride + nei2]))))
            weight[nei1 * stride + nei2] = weight[nei1 * stride + nei2] - Alpha * (
                    data[k] - weight[nei1 * stride + nei2])

        nei1 = int(winner / stride) - 1
        nei2 = (winner % stride) + 1
        if Availability(nei1, nei2, 2, stride) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[nei1 * stride + nei2]))))
            weight[nei1 * stride + nei2] = weight[nei1 * stride + nei2] - Alpha * (
                    data[k] - weight[nei1 * stride + nei2])

        nei1 = int(winner / stride) + 1
        nei2 = (winner % stride) - 1
        if Availability(nei1, nei2, 2, stride) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[nei1 * stride + nei2]))))
            weight[nei1 * stride + nei2] = weight[nei1 * stride + nei2] - Alpha * (
                    data[k] - weight[nei1 * stride + nei2])

        nei1 = int(winner / stride) + 1
        nei2 = (winner % stride)
        if Availability(nei1, nei2, 2, stride) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[nei1 * stride + nei2]))))
            weight[nei1 * stride + nei2] = weight[nei1 * stride + nei2] - Alpha * (
                    data[k] - weight[nei1 * stride + nei2])

        nei1 = int(winner / stride) + 1
        nei2 = (winner % stride) + 1
        if Availability(nei1, nei2, 2, stride) == True:
            change[k] = max(change[k], np.max(abs(Alpha * (data[k] - weight[nei1 * stride + nei2]))))
            weight[nei1 * stride + nei2] = weight[nei1 * stride + nei2] - Alpha * (
                    data[k] - weight[nei1 * stride + nei2])
    return change, weight


def SOM_fit(R, Rect, data, initial_weight, stride, num_n):
    change = np.ones(data.shape[0], np.double)
    maxchange = 1
    Alpha = 0.6
    weight = initial_weight
    D = np.zeros(num_n, np.double)
    while maxchange > 0.0001:
        for k in range(data.shape[0]):
            for j in range(num_n):
                D[j] = np.sum((weight[j] - data[k]) ** 2)
            change, weight = Update(R, Rect, k, change, Alpha, data, weight, np.argmin(D), stride, num_n)
        Alpha = 0.5 * Alpha
        maxchange = np.max(change)
        # print(maxchange)

    return weight


def SOM_Select(data, weight, labels, num_n):
    neurons = np.zeros((num_n, 10), np.double)
    winners = np.zeros(20, np.int)
    D = np.zeros(num_n, np.double)
    result = np.zeros((10, 20), int)
    for k in range(data.shape[0]):
        for j in range(num_n):
            D[j] = np.sum((weight[j] - data[k]) ** 2)
        mini = np.argmin(D)
        # print(type(mini))
        ind = labels[k].astype(int)
        neurons[mini][ind] += 1
    print('-----------------')
    for i in range(20):
        winners[i] = np.argmax(np.sum(neurons, axis=1))
        for j in range(10):
            result[j][i] = int(neurons[winners[i]][j])
            print(int(neurons[winners[i]][j]))
            neurons[winners[i]][j] = 0
        print('*******')
    print(result)
    plt.imshow(result, cmap='gray')
    plt.show()


num_n = 841
stride = 29
R = 1

data = np.loadtxt('fashion_mnist_data.csv', delimiter=',')
labels = np.loadtxt('fashion_mnist_label.csv', delimiter=',')
labels.astype(int)
print(type(labels[0]))
# y = np.loadtxt('labels.csv', delimiter=',')
# normalize data
data_normed = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
initial_weight = np.zeros((num_n, 784), np.double)
initial_weight = 0.1 * np.random.randn(num_n, 784)
weight = np.zeros((num_n, 784), np.double)

#part one
weight = SOM_fit(0, 0, data, initial_weight, stride, num_n)
SOM_Select(data, weight, labels, num_n)

print("*****************************")

#part two
weight = SOM_fit(1, 0, data, initial_weight, stride, num_n)
SOM_Select(data, weight, labels, num_n)
print("*****************************")
#part three
weight = SOM_fit(1, 1, data, initial_weight, stride, num_n)
SOM_Select(data, weight, labels, num_n)
