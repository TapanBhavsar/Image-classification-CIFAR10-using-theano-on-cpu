from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
from PIL import Image
import glob
import pickle

# this is the class names of images used in CIFAR10 dataset
class_name = ["aeroplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#read image in any (height, width)
img = Image.open('<Path>/sample.jpg')# give path of your immage for prediction
# dimensions are (height, width, channel)
new_img = img.resize((32,32))
new_img = np.asarray(new_img, dtype='float32') / 256.
# put image in 4D tensor of shape (1, height, width, 3)
img_ = new_img.reshape(-1, 32, 32, 3).transpose(0,3,1,2)

# this function gives hiher probabilistic class name
def classify_name(predicts):
    max =predicts[0,0]
    temp =0
    for i in range(len(predicts[0])):
        if predicts[0,i]>max:
            max = predicts[0,i]
            temp = i;
    print(class_name[temp])

# this function gives tuple data in form of training set, validation set and test set
def load_dataset(dirpath='<your choosen path >/cifar-10-batches-py'):#give path where you have extracted dataset file
    X, y = [], []
    for path in glob.glob('%s/data_batch_*' % dirpath):
        with open(path, 'rb') as f:
            batch = pickle.load(f)#on some system gives error about encoding use this (f,encoding='latin1')
        X.append(batch['data'])
        y.append(batch['labels'])

    #combine all mini batches data into one batch    
    X = np.concatenate(X) /np.float32(255)
    y = np.concatenate(y).astype(np.int32)
    
    # separate RGB data
    X = np.dstack((X[:, :1024], X[:, 1024:2048], X[:, 2048:]))
    
    #reshape into 4D tensor
    X = X.reshape((X.shape[0], 32, 32, 3)).transpose(0,3,1,2)
    
    # split into training and validation sets
    X_train = X[-45000:]
    y_train = y[-45000:]
    X_valid = X[:-45000]
    y_valid = y[:-45000]
    
    # load test set
    path = '%s/test_batch' % dirpath
    with open(path, 'rb') as f:
        batch = pickle.load(f)#on some system gives error about encoding use this (f,encoding='latin1')
    X_test = batch['data'] /np.float32(255)
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))
    X_test = X_test.reshape((X_test.shape[0], 32, 32, 3)).transpose(0,3,1,2)
    y_test = np.array(batch['labels'], dtype=np.int32)
    
    # normalize to zero mean and unity variance
    offset = np.mean(X_train, 0)
    scale = np.std(X_train, 0).clip(min=1)
    X_train = (X_train - offset) / scale
    X_valid = (X_valid - offset) / scale
    X_test = (X_test - offset) / scale
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

#Build convolution neural network (CNN)
def build_model(input_var=0):
    
    # this line represent input layer
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),       
                           input_var=input_var)
    
    # Convolutional layer with 32 kernels of size 5x5.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotNormal('relu'))
    
    # Max-pooling layer of factor 2 in both dimensions that convert 32X32 channels to 16x16 channels
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotNormal('relu'))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # another convolution with 64 4x4 kernels, and another 2x2 pooling
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotNormal('relu'))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 64 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=64,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

# this function divide input images and targets depends on batch size
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
# this is main function, in this function the CNN will be train, test , get accuracy on training and testing and also amke prediction on random images
def main(model='cnn', num_epochs=150):
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    network = build_model(input_var)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    #for testing perpose
    get_preds = theano.function([input_var], test_prediction)
    
    #if pretrained data is not exist
    if not os.path.exists('<path>/model.npz'):#give path for checking is your network pretrained or not , if not so it will trian the data and complete the whole process
        # Load the dataset
        print("Loading data...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

        # Finally, launch the training loop.
        print("Starting training...")
        
        # We iterate over epochs:
        for epoch in range(num_epochs):
        
        # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train,500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
        
        # Optionally, you could now dump the network weights to a file like this:
        np.savez('<Path>/model.npz', *lasagne.layers.get_all_param_values(network))#give path for where do you want to save your trained data
    
    
    else:
        # And load them again later on like this:
        with np.load('<path>/model.npz') as f:# give path of your trained model file for loading weights
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)   
        
    predict = get_preds(img_)
    classify_name(predict)
 
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)