import string # from some string manipulation tasks
import nltk # natural language toolkit
import re # regex
import numpy as np
import pandas as pd
from string import punctuation # solving punctuation problems
from nltk.corpus import stopwords # stop words in sentences
from nltk.stem import WordNetLemmatizer # For stemming the sentence
from nltk.stem import SnowballStemmer # For stemming the sentence
from contractions import contractions_dict # to solve contractions
from autocorrect import Speller #correcting the spellings
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

#Libraries for general purpose
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess
import conf_matrix


#Data preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('cyberbullying_tweets.csv')
df.drop(df[df['cyberbullying_type'] == 'other_cyberbullying'].index, inplace = True)
print(df['cyberbullying_type'].value_counts())
df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})
sentiments = ["religion","age","gender","ethnicity","not bullying"]
#converting categories to numbers
df["sentiment"].replace({"religion": 1, "age": 2, "gender": 3, "ethnicity": 4, "not_cyberbullying": 5}, inplace=True)


#preprocess tweets
texts_cleaned = []
for t in df.text:
    texts_cleaned.append(preprocess.preprocessText(t))
df['text_clean'] = texts_cleaned
df.drop_duplicates("text_clean", inplace=True)

#removing very short or very long tweets

text_len = []
for text in df.text_clean:
    tweet_len = len(text.split())
    text_len.append(tweet_len)
    
df['tweet_len'] = text_len
df = df[df['tweet_len'] > 3]
df = df[df['tweet_len'] < 100]    
    
    
print(df['sentiment'].value_counts())

#feature selection
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


tfidf = TfidfTransformer()
clf = CountVectorizer()

X_cv =  clf.fit_transform(df['text_clean'])

tf_transformer = TfidfTransformer(use_idf=True).fit(X_cv)
X_tf = tf_transformer.transform(X_cv)

#splitting the data
from sklearn.model_selection import train_test_split
# train and test
X_train, X_test, y_train, y_test = train_test_split(X_tf, df['sentiment'], test_size=0.20, stratify=df['sentiment'], random_state=42)
print(y_train.value_counts())

from imblearn.over_sampling import SMOTE
vc = y_train.value_counts()

while (vc[1] != vc[4]) or (vc[1] !=  vc[2]) or (vc[1] !=  vc[3]) :
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)
    vc = y_train.value_counts()
print(y_train.value_counts())

#training 
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

#evaluation
from sklearn.metrics import classification_report,confusion_matrix
svm_pred = svm_clf.predict(X_test)
print("\n")
print('Classification Report for Support Vector Machine:\n\n',classification_report(y_test, svm_pred, target_names=sentiments))
cm = confusion_matrix(y_test,svm_pred)
conf_matrix.print_confusion_matrix(cm,sentiments)
