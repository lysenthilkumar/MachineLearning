import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    k_classes=np.unique(y)
    means=np.zeros([X.shape[1],k_classes.size])
    #print(k_classes.size)
    covmat=[]
    count=0
    for i in np.nditer(k_classes):
        index=np.where(y==i)[0]
        required_x=X[index]
        mean_row=np.mean(required_x,axis=0)
        #print(mean_row.shape)
        #print(means.shape)
        means[:,count]=mean_row
        count=count+1
    covmat=np.cov(X.transpose())
    #print(covmat.shape)
    #print(means.shape)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    #print(X.shape)
    #print(y.shape)
    #print(y)
    #All outputs that are present(k classes)
    k_classes=np.unique(y)
    #print(k_classes.size)
    means=np.zeros([X.shape[1],k_classes.size])
    #covmats=np.zeros([X.shape[1],X.shape[1]])
    covmats=[]
    #print(means.shape)
    count=0
    for i in np.nditer(k_classes):
        index=np.where(y==i)[0]
        #print(index)
        required_x=X[index]
        #print(X[index])
        #print(required_x)
        mean_row=np.mean(required_x,axis=0)  #apply down the column
        #np.concatenate((means, mean_row), axis=0)
        means[:,count]=mean_row
        count=count+1
        current_covmats=np.cov(required_x.transpose())
        covmats.append(current_covmats)
        #print(covmats)
    # IMPLEMENT THIS METHOD

    #Find the number of output classes first
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model 2x5 & 2x2
    # Xtest - a N x d matrix with each row corresponding to a test example 100x2
    # ytest - a N x 1 column vector indicating the labels for each test example 100x1
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    brackets=np.zeros([5,1])
    ypred=np.zeros([Xtest.shape[0],1])
    outcount=0
    for j in Xtest:
        #print(j)
        count=0
        for i in means.transpose():
            #print(i)
            third=np.subtract(j,i)
            #print(third.shape)
            first=(np.subtract(j,i)).transpose()
            #firs=np.transpose(first)
            #print(firs)
            second=covmat
            #k=np.dot(first,second)
            #print(k.shape)
            #z=np.dot(k,third)
            #print(z.shape)
            mul=np.dot(np.dot(first,second),third)
            k=np.exp(-1*0.5*mul)
            #print(k)
            brackets[count]=np.exp(-1*0.5*mul)
            #print(brackets[count])
            count=count+1
        max=np.argmax(brackets)
        max=max+1
        ypred[outcount]=max
        outcount=outcount+1
    #print(ypred.shape)
    #print(ytest.shape)

    acc=0
    for check in range(0,Xtest.shape[0]):
        if(ytest[check]==ypred[check]):
            acc=acc+1
    print(acc)

        #print(brackets.shape)
            #divisor=(np.pi*2)
        #print(brackets)
    #print(mul.shape)

    # IMPLEMENT THIS METHOD
    #print(means)
    #print(covmat)
    #print(Xtest)
    #print(ytest)
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    brackets=np.zeros([5,1])
    ypred=np.zeros([Xtest.shape[0],1])
    outcount=0
    for j in Xtest:
        #print(j)
        count=0
        for i in means.transpose():
            #print(i)
            third=np.subtract(j,i)
            #print(third.shape)
            first=(np.subtract(j,i)).transpose()
            #firs=np.transpose(first)
            #print(firs)
            second=covmats[count]
            det=np.linalg.det(covmats[count])
            inv=np.linalg.inv(covmats[count])
            #k=np.dot(first,second)
            #print(k.shape)
            #z=np.dot(k,third)
            #print(z.shape)
            mul=np.dot(np.dot(first,inv),third)
            #k=np.exp(-1*0.5*mul)
            #print(k)
            brackets[count]=(np.exp(-1*0.5*mul))/(np.power(det,0.5))
            #print(brackets[count])
            count=count+1
        max=np.argmax(brackets)
        max=max+1
        ypred[outcount]=max
        #print(ypred)
        outcount=outcount+1
    #print(ypred.shape)
    #print(ytest.shape)

    acc=0
    for check in range(0,Xtest.shape[0]):
        if(ytest[check]==ypred[check]):
            acc=acc+1
    #print(acc)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    first=np.linalg.inv(np.dot(X.transpose(),X))
    second=np.dot(X.transpose(),y)
    #print(first.shape)
    #print(second.shape)
    w=np.dot(first,second)
    # IMPLEMENT THIS METHOD 
    #Might need to add a column of 1s to X                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1      
    first=np.linalg.inv((lambd*np.identity(X.shape[1]))+np.dot(X.transpose(),X))
    #second=np.dot(X.transpose(),y)
    second=np.dot(first,X.transpose())
    w=np.dot(second,y)
    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # mse
    #mse=(np.sum(np.square(np.subtract(ytest,np.dot(Xtest,w)))))/y.shape[0];
    #mse=(np.dot(np.subtract(ytest,np.dot(Xtest,w)).transpose(),
    mse=np.sum(np.square(np.subtract(ytest,np.dot(Xtest,w))))/ytest.shape[0];
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda      
    #print(w.shape)
    #print(y.shape)
    #print(X.shape)
    #s=w.shape[0]
    #print(s)
    w=w.reshape(65,1) #for subtraction         
    error=(0.5*np.sum(np.square(y-np.dot(X,w))))+((0.5*lambd)*np.dot(w.transpose(),w))
    #error_grad=(-1*np.dot(X.transpose(),np.subtract(y,np.dot(X,w))))+(lambd*w);
    #error_grad=(np.dot(w.transpose(),np.dot(X.transpose(),X)))-(np.dot(y.transpose(),X))+(lambd*w).flatten();
    first=np.dot(X.transpose(),X)
    #print(first.shape)
    second=np.dot(first,w)
    #print(second.shape)
    third=np.dot(X.transpose(),y)
    #third=np.dot(y.transpose(),X)
    #print(third.shape)
    fourth=second-third
    #print(fourth.shape)
    fifth=fourth+(lambd*w)
    #print(fifth.shape)
    error_grad=fifth.flatten()
    #error_grad = (b - c + d).flatten()
    #error_grad=0;
    #error_grad=error_grad.flatten();
    #what is error grad?
    #print(first.shape)                                                   

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
    Xd=np.zeros([x.shape[0],p+1])
    for i in range(0,p+1):
        Xd[:,i]=np.power(x,i)
    #print(Xd.shape);
    # IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y) #remove
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest) #remove
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y) #remove
qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest) #remove
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1))) 
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest) 

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest) 

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show() 
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0

mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()



# Problem 5
pmax = 7

#Group37 Code:
error_test5_0 = 9999999
error_test5_1 = 9999999

lambda_opt = mses3.argmin() * 0.01 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    
    '''
    #Group37 Code:
    if(error_test5_0 > mses5[p, 0]):
        error_test5_0 = mses5[p, 0]
        p_optimal0 = p
    '''
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    '''
    #Group37 Code: 
    if(error_test5_1 > mses5[p, 1]):
        error_test5_1 = mses5[p, 1]
        p_optimal1 = p
    '''
fig = plt.figure(figsize=[12,6])
#print ("Optimal P : ",(mses5[:,0].argmin() * 1)," ",(mses5[:,1].argmin() * 1))
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
