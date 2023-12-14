#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Downloaded stopwords file path
stopwords_file_path = 'C:\\Users\\theja\\Desktop\\New folder\\english'


# Load stopwords from the local file
with open(stopwords_file_path, 'r') as file:
    stopwords_list = file.read().splitlines()

# Load the dataset
df = pd.read_csv('spam.csv', header=None)

# Assuming the text data is in the first column (index 0)
corpus = []

for i in range(len(df)):
    review = df.iloc[i, 0]
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word.lower() not in set(stopwords_list)]
    review = ' '.join(review)
    corpus.append(review)

# Assuming the labels are in the last column (index -1)
labels = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)

# Feature extraction with CountVectorizer
cv = CountVectorizer(min_df=1)  # min_df=1 ensures that at least one word is retained
X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)

# Term Frequency-Inverse Document Frequency (TF-IDF) transformation
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)
X_test_tfidf = tfidf.transform(X_test_counts)

# Training the model
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




