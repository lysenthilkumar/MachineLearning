import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#import pickle


def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    #print(initialWeights.shape)
    initialWeights = initialWeights.reshape((n_features + 1),1)
    #print(initialWeights.shape)
    #print(train_data.shape)
    bias_matrix = np.ones((n_data,1))
    #print(bias_matrix)
    new_train_data = np.append(bias_matrix,train_data,axis=1)
    #new_train_data = np.hstack((np.ones((n_data,1)),train_data))
    #print(bias_matrix.shape)
    prob = sigmoid(np.dot(new_train_data,initialWeights))
    #print(prob.shape)
    error = (np.sum((labeli*np.log(prob))+((1.0-labeli)*np.log(1.0-prob)))*-1)/n_data
    #print(error)
    #error_grad = (np.sum((prob-labeli)*new_train_data))/n_data
    error_grad = np.sum(((prob-labeli)*new_train_data),axis=0)/n_data
    #print(error_grad)

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    bias_matrix = np.ones((data.shape[0],1))
    #print(bias_matrix)
    #print(bias_matrix.shape)
    #print(data.shape)
    new_train_data = np.append(bias_matrix,data,axis=1)
    #new_train_data = np.hstack((np.ones((data.shape[0],1)),data))
    #print(new_train_data[:0])
    prob = sigmoid(np.dot(new_train_data,W))
    label = np.argmax(prob,axis=1)
    label = label.reshape((data.shape[0],1))
    #print(label)

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #print(params.shape)
    bias_matrix = np.ones((train_data.shape[0],1))
    new_train_data = np.append(bias_matrix,train_data,axis=1)
    initialWeights = params.reshape((n_feature+1,n_class))
    first = np.dot(new_train_data,initialWeights)
    second = np.exp(first)
    third = np.sum(second,axis=1)
    #print(second.shape)
    #print(third.shape)
    third = third.reshape(third.shape[0],1)
    #print(third.shape)
    prob = second/third 
    error = ((np.sum(labeli*np.log(prob)))*-1)/n_data
    #error = ((np.sum(labeli*np.log(prob)))*-1)
    #print("Error : "+str(error))

    error_grad = (np.dot(np.subtract(prob,labeli).transpose(),new_train_data).transpose())/n_data
    #error_grad = (np.dot(np.subtract(prob,labeli).transpose(),new_train_data).transpose())
    #errror_grad = error_grad.transpose()
    error_grad = error_grad.flatten()

    #print(error_grad.shape)

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    
    bias_matrix = np.ones((data.shape[0],1))
    new_train_data = np.append(bias_matrix,data,axis=1)
    first = np.dot(new_train_data,W)
    second = np.exp(first)
    third = np.sum(second,axis=1)
    third = third.reshape(second.shape[0],1)
    prob = second/third 
    label = np.zeros((data.shape[0], 1))
    label = np.argmax(prob,axis=1)
    label = label.reshape((data.shape[0],1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

'''
with open('params.pickle','wb') as f1:
    pickle.dump(W, f1)

with open('params_bonus.pickle','wb') as f2:
    pickle.dump(W_b, f2)                        
'''

"""
Script for Support Vector Machine
"""

print('\n\n----------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

train_label = train_label.ravel()

print('######################## CODE FOR Kernel = linear ###############################')

clf = SVC(kernel='linear')
clf.fit(train_data,train_label)

#Train data
trainPredict=clf.predict(train_data)
accuracyTrain = accuracy_score(trainPredict, train_label)*100
print('Training Accuracy')
print(accuracyTrain)

#Validation Data
validationPredict=clf.predict(validation_data)
accuracyValidation=accuracy_score(validationPredict,validation_label)*100
print('Validation Accuracy')
print(accuracyValidation)

#Test Data
testPredict=clf.predict(test_data)
accuracyTest=accuracy_score(testPredict,test_label)*100
print('Test Accuracy')
print(accuracyTest)

print('######################### CODE FOR GAMMA = 1 ############################')

clf = SVC(kernel = 'rbf', gamma=1)
print("Set parameters")
clf.fit(train_data,train_label)
print("Fitting done")

#Train data
trainPredict2 = clf.predict(train_data)
print("Train Data Predict done")
accuracyTrain2 = accuracy_score(trainPredict2,train_label)*100
print('Train Accuracy')
print(accuracyTrain2)

#Validation Data
validationPredict2=clf.predict(validation_data)
print("Validation Predict done")
accuracyValidation2=accuracy_score(validationPredict2,validation_label)*100
print('Validation Accuracy')
print(accuracyValidation2)

#Test Data
testPredict2=clf.predict(test_data)
print("Test Predict done")
accuracyTest2=accuracy_score(testPredict2,test_label)*100
print('Test Accuracy')
print(accuracyTest2)


print('######################### Gamma = Default #################################')

clf = SVC(kernel='rbf')
clf.fit(train_data,train_label)

#Train data
trainPredict3 = clf.predict(train_data)
print("Training Accuracy Done")
accuracyTrain3 = accuracy_score(trainPredict3,train_label)*100
print("Train Accuracy")
print(accuracyTrain3)

#Validation Data
validationPredict3=clf.predict(validation_data)
print("Validation Predict done")
accuracyValidation3=accuracy_score(validationPredict3,validation_label)*100
print('Validation Accuracy')
print(accuracyValidation3)

#Test Data
testPredict3=clf.predict(test_data)
print("Test Predict done")
accuracyTest3=accuracy_score(testPredict3,test_label)*100
print('Test Accuracy')
print(accuracyTest3)


print('####################### FOR MULTIPLE C VALUES ########################')

cValues = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for cValue in cValues:
    clf = SVC(C = cValue, kernel='rbf')

    clf.fit(train_data,train_label)

    #Train Data
    trainPredict4 = clf.predict(train_data)
    print("Training Data predict")
    accuracyTrain4 = accuracy_score(trainPredict4, train_label)*100
    print("Train Accuracy")
    print(accuracyTrain4)

    # Validation Data
    validationPredict4 = clf.predict(validation_data)
    print("Validation Data predict")
    accuracyValidation4 = accuracy_score(validationPredict4, validation_label) * 100
    print('Validation Accuracy')
    print(accuracyValidation4)

    # Test Data
    testPredict4 = clf.predict(test_data)
    print("Test Data predict")
    accuracyTest4 = accuracy_score(testPredict4, test_label) * 100
    print('Test Accuracy')
    print(accuracyTest4)

