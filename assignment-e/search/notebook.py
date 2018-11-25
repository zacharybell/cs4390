#%% helper functions
import os, sys
import numpy as np

# from irStub.py
def valueOfSuggestion(result, position, targets):   #-----------------------------
    weight = [1.0, .5, .25]
    if result in targets:
        return weight[max(position, targets.index(result))]
    else:
        return 0


def scoreResults(results, targets):   #-----------------------------
    merits = [valueOfSuggestion(results[i], i, targets) for i in [0,1,2]]
    return sum(merits)


def parseAlternatingLinesFile(file):     #-----------------------------
   # read a sequence of pairs of lines, e.g. text of webpage(s), name/URL
   sequenceA = []
   sequenceB = [] 
   fp = open(file, 'r')
   expectingA = True 
   for line in fp.readlines():
       if expectingA:
           sequenceA.append(line.rstrip())
           expectingA = False
       else:
           sequenceB.append(line.rstrip())
           expectingA = True
   fp.close()
   return sequenceA, sequenceB

def encode_bin(data: list, cat: set) -> np.ndarray:
    """Encodes a list into a binary representation based on the category index.

    Example:
        data = ['dog', 'camel'] and cat = ['dog', 'spider', 'camel', 'fish', 'cat'] would return [1,0,1,0,0]
    
    Args:
        data: a list of categorical data
        cat:  a set of categories that act as a reference for the encoding

    Returns:
        list: a binary encoding of the categorical data
    """
    import numpy as np

    cat = list(cat)
    enc_list = np.zeros(len(cat), dtype='uint8')
    for e in data:
        enc_list[cat.index(e)] = 1
    
    return enc_list

def encode_matrix_bin(data: list, cat: list) -> np.ndarray:
    """Encodes a numpy matrix with binary values based on the categories for each 1st order element.

    This function produces values for each row using the encode_bin function.

    Args:
        data: a list of lists of categories
        cat:  a list of catories that act as a reference for the encoding
    
    Returns:
        ndarray: a matrix of binary encodings
    """
    import numpy as np

    return np.array([ encode_bin(s, cat) for s in data ], dtype='uint8')


DATA_SET = 'assignment-e/data'
DATA_SET_FILES = {
    'documents': 'csFaculty.txt',
    'test': 'testQueries.txt',
    'train': 'trainingQueries.txt'
}


#%% load the data
import pandas as pd
import string

contents, names =  parseAlternatingLinesFile(os.path.join(DATA_SET, DATA_SET_FILES['documents']))
trans = str.maketrans('', '', string.punctuation)
contents = [ doc.translate(trans) for doc in contents ]

documents = pd.DataFrame(data={'document': contents, 'name': names})

PROFS = list(documents['name'])

# create a dataframe with one-hot encoded professors
documents = pd.get_dummies(documents, columns=['name'], prefix='', prefix_sep='')
documents['name'] = np.array(PROFS)

# create a dataframe for queries with 0's or 1's for matched professors
train_queries, train_targets = parseAlternatingLinesFile(os.path.join(DATA_SET, DATA_SET_FILES['train']))
test_queries, test_targets   = parseAlternatingLinesFile(os.path.join(DATA_SET, DATA_SET_FILES['test']))
train_targets = list(map(str.split, train_targets))
test_targets  = list(map(str.split, test_targets))
train_targets = encode_matrix_bin(train_targets, PROFS)
test_targets  = encode_matrix_bin(test_targets, PROFS)
train_targets = pd.DataFrame(train_targets, columns=PROFS)
test_targets  = pd.DataFrame(test_targets, columns=PROFS)
train_df = pd.DataFrame(data={'query': train_queries})
test_df  = pd.DataFrame(data={'query': test_queries})
train_df = pd.concat([train_df, train_targets], axis=1, sort=False)
test_df  = pd.concat([test_df, test_targets], axis=1, sort=False)

#%% Unigram
from sklearn.feature_extraction.text import CountVectorizer

unigram_vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')

unigram_docs  = unigram_vectorizer.fit_transform(documents['document'])
unigram_train = unigram_vectorizer.transform(train_df['query'])

print(train_df[PROFS].head())

# for q in unigram_train:
# print(unigram_docs[5].dot(unigram_train[0]).as)
def unigram_score(docs, query):
    scores = []
    for d in docs:
        scores.append(np.sum(np.dot(d, query.T))/np.sum(d))
    return np.array(scores)

def softmax(vec: np.ndarray) -> np.ndarray:
    return np.exp(vec) / np.sum(np.exp(vec))

def unigram_pred_prob(docs, queries):
    prob_pred = []
    for q in queries:
        prob_pred.append(softmax(unigram_score(docs, q)))
    return np.array(prob_pred)

# def compute_score(unigram_docs, unigram_train, labels, ref=PROFS, threshold=0.0595):
#     import numpy as np
#     unigram_prob = unigram_pred_prob(unigram_docs, unigram_train)
#     for i, prob in enumerate(unigram_prob):
#         pred = np.ma.array(ref, mask=np.array([ int(threshold < v) for v in prob ]))
#         actu = labels[i]
#     print('score ', scoreResults(pred, actu))

# compute_score(unigram_docs, unigram_train, train_df[PROFS])

uni_prob = unigram_pred_prob(unigram_docs, unigram_train)[0]
print([ int(0.0595 < v) for v in uni_prob ])