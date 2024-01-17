# for data preprocess
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  13 17:45:49 2024

@author: yunfei Wang
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
from nltk.corpus import stopwords

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

data_obs()

# prediction classes distribution
def create_distribution(dataFile, ax):
    return sb.countplot(x='Label', data=dataFile, palette='hls', ax=ax)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
create_distribution(train_news, axs[0])
axs[0].set_title('Train Data')
create_distribution(test_news, axs[1])
axs[1].set_title('Test Data')
create_distribution(valid_news, axs[2])
axs[2].set_title('Validation Data')

plt.tight_layout()
plt.show()


# class distribution visualize
# create_distribution(train_news)
# plt.show()
# create_distribution(test_news)
# plt.show()
# create_distribution(valid_news)
# plt.show()
# test and valid data seems to be fairly evenly distributed between the classes

# testing
print("successfully show distribution")