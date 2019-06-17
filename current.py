import numpy as np
import matplotlib.pyplot as plt
import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('all_book_titles.txt')]

stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application', 'approach',
    'card', 'access', 'package', 'plus', 'etext', 'brief', 'vol',
    'fundamental', 'guide', 'essential', 'printed', 'third',
    'second', 'fourth',
})


def tokenize(s):
    tokenized = nltk.tokenize.word_tokenize(s.lower())
    tokenized = [t for t in tokenized if len(t) > 2]
    tokenized = [wordnet_lemmatizer.lemmatize(t) for t in tokenized]
    tokenized = [t for t in tokenized if t not in stopwords]
    tokenized = [t for t in tokenized if not any(c.isdigit() for c in t)]
    return tokenized


# Create word dictionary
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []

# test = tokenize(titles[0])

for title in titles:
    try:
        title_e = title#.encode('ascii', 'ignore')
        all_titles.append(title_e)
        tokens = tokenize(title_e)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception:
        pass


# Take each token and create data array (which will be numbers) - Unsupervised Learning if there are no labels
def tokens_to_vector(tok):
    x = np.zeros(len(word_index_map))
    for v in tok:
        j = word_index_map[v]
        x[j] = 1
    return x


N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))
i = 0
for tokens in all_tokens:
    X[:, i] = tokens_to_vector(tokens)
    i += 1

svd = TruncatedSVD()
Z = svd.fit_transform(X)
plt.scatter(Z[:, 0], Z[:, 1])
for i in range(D):
    plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))
plt.show()
