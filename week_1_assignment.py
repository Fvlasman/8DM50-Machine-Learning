#!/usr/bin/env python
# coding: utf-8

# # Getting started with the practicals
# 
# ***These notebooks are best viewed in Jupyter. GitHub might not display all content of the notebook properly.***
# 
# ## Goal of the practical exercises
# 
# The exercises have two goals:
# 
# 1. Give you the opportunity to obtain 'hands-on' experience in implementing, training and evaluation machine learning models in Python. This experience will also help you better understand the theory covered during the lectures. 
# 
# 2. Occasionally demonstrate some 'exam-style' questions that you can use as a reference when studying for the exam. Note however that the example questions are (as the name suggests) only examples and do not constitute a complete and sufficient list of 'things that you have to learn for the exam'. You can recognize example questions as (parts of) exercises by <font color="#770a0a">this font color</font>.
# 
# For each set of exercises (one Python notebook such as this one $==$ one set of exercises) you have to submit deliverables that will then be graded and constitute 25% of the final grade. Thus, the work that you do during the practicals has double contribution towards the final grade: as 25% direct contribution and as a preparation for the exam that will define the other 65% of the grade.
# 
# ## Deliverables
# 
# For each set of exercises, you have to submit:
# 1. Python functions and/or classes (`.py` files) that implement basic functionalities (e.g. a $k$-NN classifier) and 
# 2. A *single* Python notebook that contains the experiments, visualization and answer to the questions and math problems. *Do not submit your answers as Word or PDF documents (they will not be graded)*. The submitted code and notebook should run without errors and be able to fully reproduce the reported results.
# 
# We recommend that you clone the provided notebooks (such as this one) and write your code in them. The following rubric will be used when grading the practical work:
# 
# Component  | Insufficient | Satisfactory | Excellent
# --- | --- | --- | ---
# **Code** | Missing or incomplete code structure, runs with errors, lacks documentation | Self-contained, does not result in errors, contains some documentation, can be easily used to reproduce the reported results | User-friendly, well-structured (good separation of general functionality and experiments, i.e. between `.py` files and the Pyhthon notebook), detailed documentation, optimized for speed, <s>use of a version control system (such as GitHub)</s>
# **Answers to questions** | Incorrect, does not convey understanding of the material, appears to be copied from another source | Correct, conveys good understanding of the material, description in own words | Correct, conveys excellent level of understanding, makes connections between topics
# 
# ## A word on notation
# 
# When we refer to Python variables, we will use a monospace font. For example, `X` is a Python variable that contains the data matrix. When we refer to mathematical variables, we will use the de-facto standard notation: $a$ or $\lambda$ is a scalar variable, $\boldsymbol{\mathrm{w}}$ is a vector and $\boldsymbol{\mathrm{X}}$ is a matrix (e.g. a data matrix from the example above). You should use the same notation when writing your answers and solutions.
# 
# # Two simple machine learning models
# 
# ## Preliminaries
# 
# Throughout the practical curriculum of this course, we will use the Python programming language and its ecosystem of libraries for scientific computing (such as `numpy`, `scipy`, `matplotlib`, `scikit-learn` etc). The practicals for the deep learning part of the course will use the `keras` deep learning framework. If you are not sufficiently familiar with this programming language and/or the listed libraries and packages, you are strongly advised to go over the corresponding tutorials from the ['Essential skills'](https://github.com/tueimage/essential-skills) module (the `scikit-learn` library is not covered by the tutorial, however, an extensive documentation is available [here](https://scikit-learn.org/stable/documentation.html).
# 
# In this first set of exercises, we will use two toy datasets that ship together with `scikit-learn`. 
# 
# The first dataset is named `diabetes` and contains 442 patients described with 10 features: age, sex, body mass index, average blood pressure, and six blood serum measurements. The target variable is a continuous quantitative measure of the disease (diabetes) progression one year after the baseline measurements were recorded. More information is available [here](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/descr/diabetes.rst) and [here](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html).
# 
# The second dataset is named `breast_cancer` and is a copy of the UCI ML Breast Cancer Wisconsin (Diagnostic) datasets (more infortmation is available [here](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/descr/breast_cancer.rst) and [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). The datasets contains of 569 instances represented with 30 features that are computed from a images of a fine needle aspirate of a breast mass. The features describe characteristics of the cell nuclei present in the image. Each instance is associated with a binary target variable ('malignant' or 'benign'). 
# 
# You can load the two datasets in the following way:

# In[1]:


import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer

diabetes = load_diabetes()

breast_cancer = load_breast_cancer()


# In the majority of the exercises in this course, we will use higher-level libraries and packages such as `scikit-learn` and `keras` to implement, train and evaluate machine learning models. However, the goal of this first set of exercises is to illustrate basic mathematical tools and machine learning concepts. Because of this, we will impose a restriction of only using basic `numpy` functionality. Furthermore, you should as much as possible restrict the use of for-loops (e.g. use a vector-to-matrix product instead of a for loop when appropriate).
# 
# If `X` is a 2D data matrix, we will use the convention that the rows of the matrix contain the samples (or instances) and the columns contain the features (inputs to the model). That means that a data matrix with a shape `(122, 13)` represents a dataset with 122 samples, each represented with 13 features. Similarly, if `Y` is a 2D matrix containing the targets, the rows correspond to the samples and the columns to the different targets (outputs of the model). Thus, if the shape of `Y` is `(122, 3)` that means that there are 122 samples and each sample is has 3 targets (note that in the majority of the examples we will only have a single target and thus the number of columns of `Y` will be 1).
# 
# You can obtain the data and target matrices from the two datasets in the following way:

# In[2]:


X = diabetes.data
Y = diabetes.target[:, np.newaxis]

print(X.shape)
print(Y.shape)


# If you want to only use a subset of the available features, you can obtain a reduced data matrix in the following way:

# In[3]:


# use only the fourth feature
X = diabetes.data[:, np.newaxis, 3]
print(X.shape)

# use the third, and tenth features
X = diabetes.data[:, (3,9)]
print(X.shape)


# ***Question***: Why we need to use the `np.newaxis` expression in the examples above? 
# 
# Note that in all your experiments in the exercises, you should use and independent training and testing sets. You can split the dataset into a training and testing subsets in the following way:

# ### Answer:Because the slicing creates a 1D array ( so the shape would be (441,)) as we only select one column, however, we want to keep the 2D structure of our data (shape would be (441,1)) if we want to add more columns or merge it with other columns for instance

# In[4]:


# use the fourth feature
# use the first 300 training samples for training, and the rest for testing
X_train = diabetes.data[:300, np.newaxis, 3]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, np.newaxis, 3]
y_test = diabetes.target[300:, np.newaxis]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ## Exercises
# 
# ### Linear regression
# 
# Implement training and evaluation of a linear regression model on the diabetes dataset using only matrix multiplication, inversion and transpose operations. Report the mean squared error of the model.
# 
# To get you started we have implemented the first part of this exercise (fitting of the model) as an example.

# In[12]:


# add subfolder that contains all the function implementations
# to the system path so we can import them
import sys
sys.path.append('code/')

import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

# load the dataset
# same as before, but now we use all features
X_train = diabetes.data[:300, :]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, :]
y_test = diabetes.target[300:, np.newaxis]

# Get model weights
beta = lsq(X_train, y_train)

# Extract the bias term from model weights
w0 = beta[0]

# Determine the other model weights
w_other = beta[1:,]

# Compute the predicted model values
y_pred = w0 + np.dot(X_test,w_other)

# Compute the MSE
E = np.mean((y_test-y_pred)**2)

print(E)


# ### Weighted linear regression
# 
# Assume that in the dataset that you use to train a linear regression model, there are identical versions of some samples. This problem can be reformulated to a weighted linear regression problem where the matrices $\boldsymbol{\mathrm{X}}$ and $\boldsymbol{\mathrm{Y}}$ (or the vector $\boldsymbol{\mathrm{y}}$ if there is only a single target/output variable) contain only the unique data samples, and a vector $\boldsymbol{\mathrm{d}}$ is introduced that gives more weight to samples that appear multiple times in the original dataset (for example, the sample that appears 3 times has a corresponding weight of 3). 
# 
# <p><font color='#770a0a'>Derive the expression for the least-squares solution of a weighted linear regression model (note that in addition to the matrices $\boldsymbol{\mathrm{X}}$ and $\boldsymbol{\mathrm{Y}}$, the solution should include a vector of weights $\boldsymbol{\mathrm{d}}$).</font></p>

# The function that needs to be minimized is the weighted sum of squared residuals:

# $$\sum_{i=1}^{N}{d_i \cdot (y_i - X_i \cdot \beta)^2}$$

# To find the solution where the sum of squared residuals is minimum, we need to find a value for $\beta$ for which the expression has a minimum. In other words, the derivative with respect to $\beta$ has to be equal to zero:

# $$\frac{\partial}{\partial \beta}(\sum_{i=1}^{N}{d_i \cdot (y_i - X_i \cdot \beta)^2})=0$$

# Taking the derivative from the expression above gives:

# $$-2\sum_{i=1}^{N}{d_i\cdot X_i^T \cdot (y_i -X_i \cdot \beta)}=0$$

# Now, rearranging terms gives:

# $$\sum_{i=1}^{N}{d_i \cdot X_i^T \cdot X_i \cdot \beta}=\sum_{i=1}^{N}{d_i \cdot X_i^T \cdot y_i}$$

# This expression can be rewritten as:

# $$X^T \cdot D \cdot X\beta = X^T \cdot D \cdot y$$

# Solving for $\beta$:

# $$\beta = (X^T \cdot D \cdot X)^{-1}X^T \cdot D \cdot y$$

# ### $k$-NN classification
# 
# Implement a $k$-Nearest neighbors classifier from scratch in Python using only basic matrix operations with `numpy` and `scipy`. Train and evaluate the classifier on the breast cancer dataset, using all features. Show the performance of the classifier for different values of $k$ (plot the results in a graph). Note that for optimal results, you should normalize the features (e.g. to the $[0, 1]$ range or to have a zero mean and unit standard deviation).

# In[11]:


from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt

# load in data
X_train = breast_cancer.data[:300, np.newaxis, 3]
y_train = breast_cancer.target[:300, np.newaxis]
X_test = breast_cancer.data[300:, np.newaxis, 3]
y_test = breast_cancer.target[300:, np.newaxis]

# normalise feature data by putting it to a [0,1] range
X_normed_train = X_train/X_train.max(axis=0)
X_normed_test = X_test/X_test.max(axis=0)


# function to calculate the distance between test points and all training points
def distance(x_train,x_test_point,y_train):
    distances=[]
    for i in range(len(x_train)):
        dist = sqrt((x_train[i]-x_test_point)**2)
        distances.append((dist,int(y_train[i])))
    return distances

# function to identify k nearest neighbours from the test points
def get_neighbours(k,X_normed_train,X_normed_test,y_train):
    neighbours = []
    for i in range(len(X_normed_test)):
        distances = []
        dist = distance(X_normed_train,X_normed_test[i],y_train)
        dist.sort(key=lambda x: x[0])
        neighbour = dist[:k]
        neighbours.append(neighbour)
    return neighbours

# function to predict the class for the test points on basis of the classes of neighbours
def predict(neighbours):
    neighbour_labels = [neighbour[1] for neighbour in neighbours]

    label_counts = Counter(neighbour_labels)
    # Return the most common label (mode)
    return label_counts.most_common(1)[0][0]
       
# function to calculate the accuracy of the model by comparing predicted and measured classes per test point. 
def accuracy(k, X_normed_train, X_normed_test, y_train, y_test):
    correct = 0
    y_pred = []
    neighbours = get_neighbours(k, X_normed_train, X_normed_test, y_train)
    
    for i in range(len(y_test)):
        predicted_label = predict(neighbours[i])
        y_pred.append(predicted_label)
        if y_test[i] == predicted_label:
            correct += 1

    accuracy = correct / len(y_test)
    return accuracy, y_pred

    
    
# Define a range of k values
n=50
k_values = list(range(1, n + 1))

# Initialize lists to store accuracy scores
accuracy_scores = []

# Calculate accuracy for each k value
for k in k_values:
    accuracy_score, _ = accuracy(k, X_normed_train, X_normed_test, y_train, y_test)
    accuracy_scores.append(accuracy_score)

# Create a plot to visualize accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
plt.title('Accuracy vs. k for KNN')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# ### $k$-NN regression
# 
# Modify the $k$-NN implementation to do regression instead of classification. Compare the performance of the linear regression model and the $k$-NN regression model on the diabetes dataset for different values of $k$..

# ### Answer: Even high k value models show a higher MSE than the linear regression model.

# In[13]:


from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt

# load the data
X_train = diabetes.data[:300, np.newaxis, 3]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, np.newaxis, 3]
y_test = diabetes.target[300:, np.newaxis]

# normalise data using a [0,1] range
X_normed_train = X_train/X_train.max(axis=0)
X_normed_test = X_test/X_test.max(axis=0)

# get the distance between every test point and training point and return the distance and y_value
def distance(x_train,x_test_point,y_train):
    distances=[]
    for i in range(len(x_train)):
        dist = sqrt((x_train[i]-x_test_point)**2)
        distances.append((dist,int(y_train[i])))
    return distances

# calculate the k nearest neighbours and return the distances and y_values 
def get_neighbours(k,X_normed_train,X_normed_test,y_train):
    neighbours = []
    for i in range(len(X_normed_test)):
        distances = []
        dist = distance(X_normed_train,X_normed_test[i],y_train)
        dist.sort(key=lambda x: x[0])
        neighbour = dist[:k]
        neighbours.append(neighbour)
    return neighbours

# predict the mean value of the y_values of the k nearest neighbours
def predict(neighbours):
    neighbour_values = [neighbour[1] for neighbour in neighbours]
    y_pred = np.mean(neighbour_values)

    return y_pred
       
# calculate the accuracy of the model by calculating the mean squared error
def accuracy(k, X_normed_train, X_normed_test, y_train, y_test):
    
    y_pred = []
    neighbours = get_neighbours(k, X_normed_train, X_normed_test, y_train)
    
    for i in range(len(y_test)):
        predicted_value = predict(neighbours[i])
        y_pred.append(predicted_value)

    MSE = np.mean((y_test-y_pred)**2)

    return MSE, y_pred

    
    
# Define the range of k values to loop over
n=50
k_values = list(range(1, n + 1))

# list to store the MSE scores for different k-sized models
MSE_scores = []

# Calculate MSE for each k value
for k in k_values:
    MSE_score, _ = accuracy(k, X_normed_train, X_normed_test, y_train, y_test)
    
    MSE_scores.append(MSE_score)

# Create a plot to visualize accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, MSE_scores, marker='o', linestyle='-')
plt.title('MSE vs. k for KNN')
plt.xlabel('Number of Neighbors used (k)')
plt.ylabel('Mean squared error (MSE)')
plt.grid(True)
plt.show()


# ### Class-conditional probability
# 
# Compute and visualize the class-conditional probability (conditional probability where the class label is the conditional variable, i.e. $P(X = x \mid Y = y_i)$ for all features in the breast cancer dataset. Assume a Gaussian distribution.
# 
# <p><font color='#770a0a'>Based on visual analysis of the plots, which individual feature can best discriminate between the two classes? Motivate your answer.</font></p>
# 
# 

# ### Answer: There are quite a few features where the PDF is away from eachother for the two classes. Ones that stand out most are features 21, 23 and 24 as there the PDF's of both classes barely overlap.

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
breast_cancer = load_breast_cancer()
data_x = breast_cancer.data
data_y = breast_cancer.target

# Get the number of features x has
num_features = data_x.shape[1]

# Initialize subplots for plotting PDFs for every feature
fig, axs = plt.subplots(num_features, 1, figsize=(6, 2 * num_features))
fig.subplots_adjust(hspace=1.5)

for i in range(num_features):
    ax = axs[i]
    
    # Separate the data into Class 0 and Class 1 for the current feature
    x_0_feature = data_x[data_y == 0, i]
    x_1_feature = data_x[data_y == 1, i]
    
    # Calculate the mean and standard deviation for Class 0 and Class 1
    mean_x0 = np.mean(x_0_feature)
    std_x0 = np.std(x_0_feature)
    mean_x1 = np.mean(x_1_feature)
    std_x1 = np.std(x_1_feature)
    
    # Calculate the PDFs for Class 0 and Class 1 using Gaussian distribution
    x_range = np.linspace(min(data_x[:, i]) - 2 * std_x0, max(data_x[:, i]) + 2 * std_x0, 100)
    PDF_0 = (1 / (std_x0 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_x0) / std_x0) ** 2)
    PDF_1 = (1 / (std_x1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_x1) / std_x1) ** 2)

    # Plot the PDFs for both classes
    ax.plot(x_range, PDF_0, label='Class 0', color='red')
    ax.plot(x_range, PDF_1, label='Class 1', color='blue')
    ax.set_title(f'Feature {i + 1}')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Probability Density')
    ax.legend()

plt.show()


# In[ ]:




