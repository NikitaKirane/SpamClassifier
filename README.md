# SpamClassifier
Machine learning algorithm which classifies emails as spam or non-spam

Files:
1. spam_classifier.py: Main code in python
2. spamTest.mat: File has testing data set "Xtest" and it's labeled as "ytest". Xtest has data of 1000 emails and each email has 1899 features. ytest has 1000 labels. Xtest Dimension = (1000, 1899) and ytest Dimension = (1000, 1)
3. spamTrain.mat: File has training data set "X" and it's labeled as "y". X has data of 4000 emails and each email has 1899 features. y has 4000 labels. X Dimension = (4000, 1899) / y Dimension = (4000, 1)
4. vocab.txt: Our vocabulary list was selected by choosing all words which occur at least 100 times in the spam corpus, resulting in a list of 1899 words.


Data:

In our program, we have combined spamTest.mat and spamTrain.mat so the combined data has 5000 emails and each email has 1899 features. The dimension of our data is (5000, 1899) and Dimension for labels is (5000, 1)
Each email has 1899 features and the value of each feature will be either 0 or 1 depends on the presence or the absence of the words in our email.
The value of each label is either 0 (non-spam) or 1 (spam).
