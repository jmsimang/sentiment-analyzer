# Python Sentiment Analyzer

Some background: A sentiment is a measure of how positive or negative something is. It can be applied to things such as
Amazon reviews, Yelp reviews, hotel reviews, tweets, etc.

## About the program
The python program analyses electronics reviews (messages, emails, etc.) and classifies them as positive or negative,
using Logistic Regression.

## How it works
Program performs classification since the reviews datasets are already marked 'positive' and 'negative'.
The program will only look at the key 'review_text' and do 2 passes on the data, 
* 1. to determine vocabulary size and find which index corresponds to which word, and 
* 2. to create data vectors
It will finally use an SKLearn classifier to interpret the weights.

## Dataset
The data can be obtained at https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

## Use-cases
Machine Learning classification models.<br>
Analyse stock markets based on news.<br>
