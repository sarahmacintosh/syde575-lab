import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import time


data = loadmat('mnist_.mat')
train_images = 2*(data['mnist']['train_images'][0,0].astype(np.float32)/255.0 - 0.5).transpose(2,0,1)
train_labels = data['mnist']['train_labels'][0,0].flatten()
train_labels = np.eye(10)[train_labels].astype(np.float32)

val_images = (2*(data['mnist']['test_images'][0,0].astype(np.float32)/255.0 - 0.5)).transpose(2,0,1)
val_labels = data['mnist']['test_labels'][0,0].flatten()
val_labels = np.eye(10)[val_labels.flatten()].astype(np.float32)


# Show one of the images as an example
plt.gray()
plt.imshow(train_images[0])
plt.show()



def construct_MLP(lr):
    model = torch.nn.Sequential(OrderedDict([
        ('l0',torch.nn.Linear(28*28, 100)),
        ('a0',torch.nn.ReLU()),
        ('l1',torch.nn.Linear(100,100)),
        ('a1',torch.nn.ReLU()),
        ('l2',torch.nn.Linear(100,10)),
        
        ]))

    
    
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(),lr=lr, momentum=0.9)
    return model, loss, opt

def construct_CNN(lr):
    model = torch.nn.Sequential(OrderedDict([
        
        ('l0',torch.nn.Conv2d(1,8,3)),
        ('a0',torch.nn.ReLU()),
        ('mp0', torch.nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))),
         
        ('l1',torch.nn.Conv2d(8,16,3)),
        ('a1',torch.nn.ReLU()),
        ('mp1', torch.nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))),

        ('l2',torch.nn.Conv2d(16,32,3)),
        ('a2',torch.nn.ReLU()),
        ('f', torch.nn.Flatten()),
        ('l3',torch.nn.Linear(32*3*3,10)),
        ]))
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(),lr=lr, momentum=0.9)
    return model, loss, opt
    
def run_MLP(train_images, train_label, val_images, val_label, epochs, lr):
    # Model setup
    model, loss, opt = construct_MLP(lr)
    
    train_input_ = train_images.reshape(-1,28*28)
    val_input_ = val_images.reshape(-1,28*28)
    train_model(model,loss,opt,train_input_, train_label, val_input_, val_labels, epochs)

def run_CNN(train_images, train_label, val_images, val_label, epochs, lr):
    model, loss, opt = construct_CNN(lr)

    train_input_ = train_images.reshape(-1,1,28,28)
    val_input_ = val_images.reshape(-1,1,28,28)
    train_model(model,loss,opt,train_input_, train_label, val_input_, val_labels, epochs)
    
def train_model(model, loss, opt, train_input_, train_label, val_input_, val_label, epochs):
    batch_size = 128
    N = train_input_.shape[0]
    inds = np.arange(0, N)

    l_train_loss = []
    l_val_loss = []
    l_train_acc = []
    l_val_acc = []

    
    # Training loop
    t0 = time.time()
    for i in range(epochs):
        
        # Shuffle training data
        np.random.shuffle(inds)
        train_input_ = train_input_[inds]
        train_label = train_label[inds]


        # Compute batches
        # TODO:
        batches = []
        n_batches = N//batch_size + 1
        for j in range(n_batches):
            batches.append((
                torch.from_numpy(train_input_[j*batch_size:(j+1)*batch_size]),
                torch.from_numpy(train_label[j*batch_size:(j+1)*batch_size])))
        
        #print(len(batches))

        for input_, label in batches:
            # Evaluate model
            opt.zero_grad()
            y = model(input_)
            loss_ev = loss(y,label)
            
            # backpropagate gradients
            loss_ev.backward()
            opt.step()

        # evaluate progress
        all_y = model(torch.from_numpy(train_input_))
        l_train_loss.append(loss(all_y,
                                 torch.from_numpy(train_label)).detach().numpy())
        l_train_acc.append(np.sum(np.argmax(all_y.detach().numpy(),axis=1)==np.argmax(train_label,axis=1)) / all_y.shape[0])

        print('Epoch {}: training_loss={:.4f}'.format(i, l_train_loss[-1]))
              
        all_y = model(torch.from_numpy(val_input_))
        l_val_loss.append(loss(all_y,
                                 torch.from_numpy(val_label)).detach().numpy())
        l_val_acc.append(np.sum(np.argmax(all_y.detach().numpy(),axis=1)==np.argmax(val_label,axis=1)) / all_y.shape[0])
    
    t1 = time.time()
    total_time = t1 - t0
    print('Training time: {:.4f} seconds'.format( total_time))
    print('Final training accuracy {:.4f}'.format(l_train_acc[-1]))
    print('Final validation accuracy {:.4f}'.format(l_val_acc[-1]))
    
    # Plot loss per epoch
    ax0 = plt.subplot(211)
    ax0.plot(np.arange(epochs), l_train_loss, label="training")
    ax0.plot(np.arange(epochs), l_val_loss, label="validation")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    
    ax1 = plt.subplot(212)
    ax1.plot(np.arange(epochs), l_train_acc, label="training")
    ax1.plot(np.arange(epochs), l_val_acc, label="validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("accuracy")
    plt.legend()
    plt.show()




lr = 0.01
epochs = 10

run_MLP(train_images, train_labels, val_images, val_labels, epochs, lr)
run_CNN(train_images, train_labels, val_images, val_labels, epochs, lr)

