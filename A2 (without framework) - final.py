import numpy as np
import matplotlib.pyplot as plt
import time
from dataset.mnist import load_mnist

#%% functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

#%% made functions
def accuracy(x, t):
    W1, W2 = pr['W1'], pr['W2']
    b1, b2 = pr['b1'], pr['b2']

    net1 = np.dot(x, W1) + b1
    o1 = sigmoid(net1)
    net2 = np.dot(o1, W2) + b2
    y = sigmoid(net2)
    
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    
    acc = np.sum(y == t) / float(x.shape[0])
    return acc

#%% parameters

# parameters
input_size = 784
hidden_size = 300
output_size = 10

# hyperparameters
iters_num = 30000
train_size = 60000
batch_size = 300
learning_rate = 0.5

train_acc_list = []
test_acc_list = []

epoch = 0

#%% parsing
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# weight initalization - Normal Distribution (std=0.01)
pr = {}
pr['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
pr['b1'] = np.zeros(hidden_size)
pr['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
pr['b2'] = np.zeros(output_size)

#%% training
# start timer
start_time = time.time()

for i in range(iters_num):
    
    # mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x = x_train[batch_mask]
    t = t_train[batch_mask]
     
    # gradient
    W1, W2 = pr['W1'], pr['W2']
    b1, b2 = pr['b1'], pr['b2']
    
    # forwards using sigmoid
    net1 = np.dot(x, W1) + b1
    o1 = sigmoid(net1)
    net2 = np.dot(o1, W2) + b2
    y = sigmoid(net2)
    
    # backwards using SGD, sigmoid
    pr['W2'] -= learning_rate * np.dot(o1.T, (y - t) / batch_size * sigmoid_grad(net2))
    pr['b2'] -= learning_rate * np.sum((y - t) / batch_size, axis=0)
    pr['W1'] -= learning_rate * np.dot(x.T, sigmoid_grad(net1) * np.dot((y - t) / batch_size, W2.T))
    pr['b1'] -= learning_rate * np.sum(sigmoid_grad(net1) * np.dot((y - t) / batch_size, W2.T), axis=0)
        
    # accuracy
    if i % 1000 == 0:
        train_acc = accuracy(x_train, t_train)
        test_acc = accuracy(x_test, t_test)
        print("Epoch: " + str(epoch) +
              "\ttrain acc: " + str(train_acc) +
              ",\ttest acc: " + str(test_acc) +
              ",\t time lapsed: " + str(time.time() - start_time))
        start_time = time.time()
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        epoch += 1
        
#%% graphing
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train accuracy')
plt.plot(x, test_acc_list, label='test accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.legend()
plt.show()