# Module for regular expression in python
import re

# Module for tokenizing a text
from nltk.tokenize import word_tokenize

# Module for stemming words
from nltk import PorterStemmer

# Module for scientific computation in python
import numpy as np

# Module to import mat data file in python
from scipy.io import loadmat

# Module for training support vector machine
from sklearn import svm

# Module for computing accuracy
from sklearn.metrics import accuracy_score
import sys


def get_vocab_dict_function(reverse=False):
    vocab_dict = {}
    with open('vocab.txt') as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[val] = key

    return vocab_dict


def process_email_function(email):
    # converting the whole email to lower case
    email = email.lower()
    # replacing all html tags and formats to white space character
    email = re.sub(r'<.*?>', r' ', email)
    # replacing all the URLs into the text "httpaddr"
    email = re.sub(r'(https?)?://www\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+/?', r'httpaddr', email)
    # replacing all the email address with the text "emailaddr"
    email = re.sub(r'[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+', r'emailaddr', email)
    # replacing all the digits by the text "number"
    email = re.sub(r'\d+', r'number', email)
    # replacing the dollar sign by the text "dollar"
    email = re.sub(r'\$+', r'dollar', email)
    # tripping all the punctuation, white space and new line character with the white space
    email = re.sub(r'[^\w]+', r' ', email)
    #print(email)
    return email


def email_to_stemmed_word_function(email):
    stemmed_word_list = []
    email = process_email_function(email)
    # converting the complete email into tokens of single word
    tokens = word_tokenize(email)
    #print(tokens)
    ps = PorterStemmer()
    for token in tokens:
        stemmed_word_list.append(ps.stem(token))
    #print(stemmed_word_list)
    return stemmed_word_list


def email_to_indices_list_function(email, vocab_dict):
    indices_list = []
    stemmed_word_list = email_to_stemmed_word_function(email)
    for word in stemmed_word_list:
        if word in vocab_dict:
            indices_list.append(vocab_dict[word])

    return indices_list


def email_to_feature_vector_function(email):
    vocab_dict = get_vocab_dict_function()
    feature_vector = np.zeros((1, len(vocab_dict)))
    indices_list = email_to_indices_list_function(email, vocab_dict)
    for i in indices_list:
        feature_vector[0, i-1] = 1

    return feature_vector


raw_email = """How are you?

 

Summer is passed and it seems only a few DAYS since we last saw you.  Thanks again for allowing us to give

you airport transportation.  I hope your living conditions are satisfactory and your families far away are good.

I trust also your studies are progressing and you are well. 

 

My wife, Becky, and I would like to invite you to our home on 19 October for dinner and a few fun things.  We

will provide transportation to certain locations and would like very much to have you join us.  We would collect

you around 5:00 PM and return you around 10:00 PM.

 

Would you kindly reply to confirm you can come by 11 October.  If you have a new email or phone number

please let me know.  Also please provide your current address so we can update our records and know where

you are living.

 

Warm Regards

Ron

678-230-9984
"""
email = email_to_feature_vector_function(raw_email)
print(email)
#email = email.reshape(1, 1899)
# Now we will train the support vector machine on a preprocessed data set
# training set
training_data = loadmat('spamTrain.mat')
#print(training_data)
X = training_data['X'] # Dictionary's key
y = training_data['y'] # Dictionary's Value
#print(y.ravel().shape)

# testing set
testing_data = loadmat('spamTest.mat')
Xtest = testing_data['Xtest']
ytest = testing_data['ytest']


def get__gaussian_param_function(X, y, Xtest, ytest):
    C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    score = np.zeros((len(C), len(sigma)))
    for c in range(0, len(C)):
        for s in range(0, len(sigma)):
            svm_clf = svm.SVC(kernel="rbf", C=C[c], gamma=sigma[s])
            svm_clf.fit(X, y.ravel())
            score[c, s] = accuracy_score(ytest, svm_clf.predict(Xtest))
    max_C_index, max_sigma_index = np.unravel_index(score.argmax(), score.shape)

    return C[max_C_index], sigma[max_sigma_index]


def get_linear_param_function(X, y, Xtest, ytest):
    C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    score = []
    for i in range(0, len(C)):
        svm_clf = svm.SVC(kernel="linear", C=C[i])
        svm_clf.fit(X, y.ravel())
        score.append(accuracy_score(ytest, svm_clf.predict(Xtest)))
    score = np.array(score)
    #print(score)
    return C[score.argmax()]


# initialising the value for svm regularisation parameter and sigma
C = 0.03
#c=get_linear_param_function(X,y,Xtest,ytest)
# training the support vector machine using our training set
svm_clf = svm.SVC(kernel="linear", C=C)
svm_clf.fit(X, y.ravel())
print(svm_clf.predict(email))
# accuracy score of the training and the testing set
#print("Accuracy of the training set is ", accuracy_score(y, svm_clf.predict(X))*100, "%")
#print("Accuracy of the testing set is ", accuracy_score(ytest, svm_clf.predict(Xtest))*100, "%")