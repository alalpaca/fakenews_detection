# Fake News Detection Project
SHU 23-24 Pattern Recognition course

In this project, we have used various natural language processing techniques and machine learning algorithms to classify fake news articles with python. 

## Getting Started

## File descriptions
### preprocess.py
This file contains all the preprocessing functions needed to process all input documents and texts. First we read the train, test and validation data files then performed some pre processing like tokenizing, stemming etc. There are some exploratory data analysis is performed like response variable distribution and data quality checks like null or missing values etc.

### feature_selection.py
Before we can train an algorithm to classify fake news labels, we need to extract features from it. It means reducing the mass
of unstructured data into some uniform set of attributes that an algorithm can understand.  
This file contains the feature selection algorithms for fake news detection, which includes word counts (bag of words) and tf-idf.