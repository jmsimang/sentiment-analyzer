import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from data_retriever import run_data_retriever

nltk.download('punkt')
nltk.download('wordnet')

url = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz'
target = 'domain_sentiment_data.tar.gz'

# Download the dataset for the program
run_data_retriever(url, target)

# Lemmatizer - it turns words into their base forms
wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.strip() for w in open('stopwords.txt'))

# Load reviews
positive_reviews = BeautifulSoup(open('dataset/positive.review').read(), 'lxml')
positive_reviews = positive_reviews.findAll('review_text')
negative_reviews = BeautifulSoup(open('dataset/negative.review').read(), 'lxml')
negative_reviews = negative_reviews.findAll('review_text')

# Shuffle positive reviews and trim all to the same size
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

# Create word dictionary
word_index_map = {}
current_index = 0


def tokenize(s):
    tokenized = nltk.tokenize.word_tokenize(s.lower())
    tokenized = [t for t in tokenized if len(t) > 2]
    tokenized = [wordnet_lemmatizer.lemmatize(t) for t in tokenized]
    tokenized = [t for t in tokenized if t not in stopwords]
    return tokenized


positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = tokenize(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = tokenize(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


# Take each token and create data array (which will be numbers)
def tokens_to_vector(tok, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tok:
        i = word_index_map[t]
        x[i] += 1
    x /= x.sum()
    x[-1] = label
    return x


N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map) + 1))
# keep track of which sample
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i, :] = xy
    i += 1
for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i, :] = xy
    i += 1

# Shuffle data before splitting train and test
np.random.shuffle(data)

# X is all rows & everything except the last columns, Y is the last column
X = data[:, :-1]
Y = data[:, -1]
# Data split, last 100 rows will be test dataset
X_train = X[:-100, ]
y_train = Y[:-100, ]
X_test = X[-100:, ]
y_test = Y[-100:, ]

# Create the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
print('Classification rate: ', model.score(X_test, y_test))

# View weights that each word has and see if it has positive or negative sentiments
threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)
