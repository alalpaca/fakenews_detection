# for data preprocess
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  13 17:45:49 2024

@author: yunfei Wang
"""

import os
import pandas as pd
import csv
import numpy as np
import nltk
nltk.download('treebank')
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# before reading the files, set up the working directory to point to project repo
# 打印当前工作目录
current_working_directory = os.getcwd()
print("当前工作目录:", current_working_directory)


# reading data files
test_filename = 'test.csv'
train_filename = 'train.csv'
valid_filename = 'valid.csv'
# read as dataframe
train_news = pd.read_csv(train_filename, encoding='utf-8')
test_news = pd.read_csv(test_filename, encoding='utf-8')
valid_news = pd.read_csv(valid_filename, encoding='utf-8')

# testing
print("successfully read data")

# data observation
def data_obs():
    print("training dataset size:")
    print(train_news.shape)
    print(train_news.head(10))

    # print(test_news.shape)
    # print(test_news.head(10))

    print(valid_news.shape)
    print(valid_news.head(10))

    # testing
    print("successfully data_obs")

# data_obs()

# prediction classes distribution
def create_distribution(dataFile):
    return sb.countplot(x='Label', data=dataFile, palette='hls')
# test and valid data seems to be fairly evenly distributed between the classes

# testing
print("successfully show distribution")


# data integrity check
# for missing label values
def data_qualityCheck():
    print("Checking data qualitites...")
    train_news.isnull().sum()
    train_news.info()

    print("check finished.")

    # below datasets were used to
    test_news.isnull().sum()
    test_news.info()

    valid_news.isnull().sum()
    valid_news.info()

# data_qualityCheck()


nltk.download('stopwords')
eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

porter = PorterStemmer()
def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Function for stemming tokens
import nltk
nltk.download('punkt')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

# Process the data
def process_data(data, exclude_stopword=True, stem=True):
    tokens = word_tokenize(data.lower())

    # Stemming
    if stem:
        tokens = stem_tokens(tokens, eng_stemmer)

    # Exclude stopwords
    if exclude_stopword:
        tokens = [w for w in tokens if w not in stopwords]

    return tokens

# Function to create bigrams
def create_bigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 1:
        lst = []
        for i in range(Len - 1):
            for k in range(1, skip + 2):
                if i + k < Len:
                    lst.append(join_str.join([words[i], words[i + k]]))
    else:
        # set it as unigram
        lst = create_unigram(words)
    return lst

# Function to create unigrams
def create_unigram(words):
    assert type(words) == list
    return words

# Function to process data and create ngrams
def process_and_create_ngrams(data, exclude_stopword=True, stem=True):
    tokens = process_data(data, exclude_stopword, stem)
    bigrams = create_bigrams(tokens)
    return tokens + bigrams

# Preprocess the data
def preprocess_data(input_filename, output_filename):
    # Read data
    data = pd.read_csv(input_filename, encoding='utf-8')

    # Data observation
    data_obs()

    # Class distribution visualize
    create_distribution(data)
    plt.show()

    # Data integrity check
    data_qualityCheck()
    #
    # Process the data including stemming and stopwords removal
    data['text_processed'] = data['Statement'].apply(lambda x: process_and_create_ngrams(x, True, True))

    # Save the processed data to a new CSV file
    data.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"Processed data saved to {output_filename}")
"""
# Preprocess the training data
preprocess_data('train.csv', 'train_processed.csv')

# Preprocess the testing data
preprocess_data('test.csv', 'test_processed.csv')

# Preprocess the validation data
preprocess_data('valid.csv', 'valid_processed.csv')
"""

print("********************* preprocess finish ***************************")