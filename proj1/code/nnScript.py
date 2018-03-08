import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import math
#from sklearn.feature_selection import VarianceThreshold



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # return 1 / (1 + math.exp(-z));# your code here
    return 1.0 / (1.0 + np.exp(-1.0 * np.array(z)))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    # # print(train_preprocess)
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # # print(key) all test,train6 etc
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label #0-9
            # # print(label);
            tup = mat.get(key)
            # # print(tup) 0 matrices
            sap = range(tup.shape[0])
            # # print(sap) range(0,5000)
            tup_perm = np.random.permutation(sap)
            # # print(tup_perm)  [5149 3476 2761 ..., 3427 1578 2486]
            tup_len = len(tup)  # get the length of current training set
            # # print(tup_len) 5851
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            # # print(train_preprocess)
            train_len += tag_len
            # # print(train_len)

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            # # print(label)
            # # print(train_label_preprocess)
            train_label_len += tag_len
            # # print(train_label_len)

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000
            # # print(train_len)
            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    # # print(train_data)
    train_data = train_data / 255.0
    # # print(train_data)
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # # print("Train data before")
    # # print(train_data[500])
    # # print("Test data before")
    # # print(test_data[500])
    # Feature selection
    # Your code here.

    # # print("Train data before")
    # for dataaa in train_data[505:520]:
    # # # printdataaa

    common_columns = np.all(train_data == train_data[0, :], axis=0)
    #print("CC: ", common_columns)
    #print("CC.S ", common_columns.shape)
    falseIndex = np.where(common_columns == False)
    index = np.where(common_columns == True)

    #array_valid = np.concatenate(falseIndex)
    #print("Arr Valid: ",array_valid)

    array_cc = np.concatenate(index)
    #print("Array_cc: ",array_cc)

    train_data = np.delete(train_data, array_cc, axis=1)
    #print(train_data.shape)
    #print(train_data)

    test_data = np.delete(test_data, array_cc, axis=1)
    validation_data = np.delete(validation_data, array_cc, axis=1)

    # idx = [aop[0] for aop in np.where(ax==True)]
    # print(index)
    # # print(aop)
    # # print(train_data.shape)
    # u, indices = np.unique(train_data, return_index=True)
    # # print(indices)
    # # print(np.unique(train_data))
    # sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    # train_data=sel.fit_transform(train_data)
    # test_data=sel.fit_transform(test_data)
    # # print("Test data after")
    # for dataaa in train_data[505:520]:
    # # # printdataaa
    # # print(new_test_data[500])
    # # print("Train data after")
    # # print(new_train_data[500])


    #print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    # # print(training_data.shape)
    training_data1 = np.column_stack((training_data, np.ones(len(training_data))))
    # # print(training_data1.shape)
    w1t = np.transpose(w1)
    # # print("w1t.shape", w1t.shape);
    z1 = training_data1.dot(w1t)
    # print("Z1", z1.shape);
    # # print(z1.shape)
    output_hidden = sigmoid((z1))
    # # print(output_hidden.shape)
    output_hidden_bias = np.column_stack((output_hidden, np.ones(len(training_data))))
    # print("OP HIDDEN SHAOE", output_hidden.shape)
    # print("w2 shape", w2.shape)
    w2t = np.transpose(w2)
    z2 = output_hidden_bias.dot(w2t)
    output_final = sigmoid(z2)
    # print("OP FINAL SHAOE", output_final[12], output_final[123], output_final[412])

    # For output vector, all values except one value are 0
    y = np.zeros((len(training_data), n_class))

    # In output vector, set the correct value to one corresponding to the given training label
    for i in range(len(training_data)):
        y[i][int(training_label[i])] = 1

    # Back Progogation

    value_del = (y - output_final) * output_final * (1 - output_final)
    # print("del ", value_del)
    lossRHS = np.multiply(np.subtract(1.0, y), np.log(np.subtract(1.0, output_final)))
    logVal = np.multiply(y, np.log(output_final))
    # print("log val ", logVal.shape)
    RHSsum = np.add(logVal, lossRHS)
    lossMatrix = np.sum(RHSsum);

    loss = -1 * (1 / float(y.shape[0])) * lossMatrix
    # print("Loss    ", loss)
    value_del_temp = np.transpose(value_del)

    # CALCULATE delta
    delta = output_final - y;
    # print("delta shape", delta.shape)
    # # print(value_del.shape)
    grad_w2 = (1 / len(train_data)) * (np.dot(delta.transpose(), output_hidden_bias) + np.multiply(lambdaval, w2))  # Equation 9 #Have to assign to grad_w1
    ## print("GEE", grad_w1.shape)

    # CHECK EQUATION 12, I don't know how to proceed

    zj = output_hidden * (1 - output_hidden)
    w2_N_bias = np.delete(w2, len(w2.T) - 1, axis = 1)
    deltaProd = np.dot(delta, w2_N_bias)
    grad_w1 = (1 / len(train_data)) * (np.dot(np.multiply(zj, deltaProd).transpose(),
                     training_data1) + (lambdaval * w1)) # Have to assign to grad_w2

    # # print(result2.shape)
    # # print(training_data.shape)
    # result2transpose = np.transpose(result2)
    # result_final_bp = np.dot(result2transpose, training_data)

    # print("grad shape ", grad_w2.shape)
  #  grad_w1 = np.delete(grad_w1, n_hidden, 0)
    # print("grad shape 2 ", grad_w2.shape)

    # # print(output_final)

    w1Sqr = w1 * w1
    w2Sqr = w2 * w2
    regularization = (lambdaval / float(2 * (y.shape[0]))) * (np.sum(w1Sqr) + np.sum(w2Sqr))
    # print("reg ", regularization)
    obj_val = loss + regularization
    # print("Net reg ", obj_val)
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    # print(obj_grad.shape)
    # print("1 onb GRAD", obj_grad)
    # obj_grad = np.array([obj_grad])
    # print("2 onb GRAD", obj_grad)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""
    data = np.column_stack((data, np.ones(data.shape[0])))
    w1t = np.transpose(w1)
    z1 = data.dot(w1t)
    output_hidden = sigmoid((z1))
    output_hidden = np.column_stack((output_hidden, np.ones(output_hidden.shape[0])))
    w2t = np.transpose(w2)
    z2 = output_hidden.dot(w2t)
    output_final = sigmoid(z2)
    # print("OFinal", output_final[108], output_final[260], output_final[500])

    labels = np.array([])
    labels = np.argmax(output_final, axis=1)
    # print("Labels after ", labels)
    # print("Labels ", labels.shape)
    # labels = np.array([])

    # Your code here

    return labels


"""**************Neural Network Script Starts here********************************"""


train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# print("n input shape", n_input)
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
# print("W1 and W2 sha00pe", initial_w1.shape, initial_w2.shape)
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
# print("InitialWeightsShape", initialWeights.shape)
# set the regularization hyper-parameter
lambdaval=0.0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.


#Edited by Team37 to produce time details
#startTime = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#resultTime = time.time() - startTime
#print("Time taken is: " + str(time.time() - startTime) + " seconds")

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))



'''
#Team37 Pickle Generation Script
common_columns = np.all(train_data == train_data[0, :], axis=0)
falseIndex = np.where(common_columns == False)
index = np.where(common_columns == True)
array_valid = np.concatenate(falseIndex)
#print("The output is:  ",array_valid)
obj = [array_valid, n_hidden, w1, w2, lambdaval]
# selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
pickle.dump(obj, open('params.pickle', 'wb'))
'''




# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset
# print('\n Training set Accuracy:' + str(100 * float(np.mean((predicted_label == train_label))))+ '%')
# print(predicted_label.shape, train_label.shape)
# print(train_label)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


'''
#Team37 --- Delete after Graph build. -----------------------------------------------------------------------------@#$

predicted_label1 = nnPredict(w1, w2, train_data)
trainingAccuracy = 100 * (np.mean((predicted_label1 == train_label).astype(float)))
#print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label2 = nnPredict(w1, w2, validation_data)
validationAccuracy = 100 * np.mean((predicted_label2 == validation_label).astype(float))
#print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label2 == validation_label).astype(float))) + '%')

predicted_label3 = nnPredict(w1, w2, test_data)
testAccuracy = 100 * np.mean((predicted_label3 == test_label).astype(float))
#print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label3 == test_label).astype(float))) + '%')

print (str(lambdaval)+","+str(resultTime)+","+str(trainingAccuracy)+","+str(validationAccuracy)+","+str(testAccuracy))
'''