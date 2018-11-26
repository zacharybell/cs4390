import fnmatch
import numpy as np
import pandas as pd
import pickle
import os
import scipy as sci

from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer


CORNELL_DATA = 'assignment-d/data/cornell'
IMDB_DATASET = 'imdb.npz'
RATERS       = ['Dennis+Schwartz', 'James+Berardinelli', 'Scott+Renshaw', 'Steve+Rhodes']
PANDAS       = 'assignment-d/data/processed.h5'
PICKLE       = 'assignment-d/data/processed.pkl'


def tokenize(text: np.ndarray):
    """Tokenizes an array of documents.

    Args:
        text : a numpy array containing text to be tokenized. 
    
    Return:
        A numpy matrix with each row containing an array of tokenized words from each document.
    """
    
    tokenizer = TweetTokenizer()
    return np.array([ tokenizer.tokenize(row) for row in text ])


def add_padding(sparse_matrix: np.ndarray, dtype: np.dtype) -> np.ndarray:   
    """Pads a sparse matrix with empty elements.

    Args:
        sparse_matrix (numpy.ndarray) : the sparse matrix to be padded
        dtype (numpy.dtype) : a numpy datatype that defines the datatype of the returned matrix

    Returns:
        A padded regular numpy matrix.
    """

    max_length_row = max([ len(row) for row in sparse_matrix ])
    
    padded_matrix = np.zeros((len(sparse_matrix), max_length_row), dtype=dtype)
    for i, row in enumerate(sparse_matrix):
        padded_matrix[i, :len(row)] = row
    return padded_matrix


def feature_label_pair(path, feature_reg, label_reg):
    """Finds and returns a relative path for a feature and label file.
    
    This function is intended to locate feature label pairs in a directory.
    
    Args:
        path (str)      : the path to the searched directory
        feature_reg(str): a regex for the feature file
        label_reg (str) : a regex for the label file
        
    Returns:
        A tuple of feature file's path and label's file path or None if one (or both)
        aren't located.        
    """
    feature_file = ''
    label_file = ''
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, feature_reg):
            feature_file = os.path.join(path, file)
        if fnmatch.fnmatch(file, label_reg):
            label_file = os.path.join(path, file)
        if feature_file and label_file:
            return (feature_file, label_file)
    return


def embed_transform(data, min_count=1, size=100):
    """Transforms a matrix of words into word embeddings.

    Each word in the matrix will be converted to a word embedding. A matrix of MxN will become MxNxC where C is the 
    size of the word vector.

    Args:
        data (numpy.ndarray) : a matrix containing words
        min_count (int)      : the minimum number of words needed to train the embedding model
        size (int)           : the size of the word embedding vectors produced 

    Returns:
        A matrix with all of the words represented with word embeddings.
    """

    w2v = Word2Vec(sentences=data.tolist(), min_count=min_count, size=size)
    embed_data = np.zeros(data.shape+(size,))

    def sent_to_vec(model, sentence):
        return np.array([model.wv[w] for w in sentence])

    for i, sentence in enumerate(data):
        embed_data[i,:] = sent_to_vec(w2v, sentence)

    return embed_data


## Load data files into dataframe
cornell_df   = pd.DataFrame()
for r in RATERS:
    (ff, lf)          = feature_label_pair(os.path.join(CORNELL_DATA, r), 'subj*', 'rating*')
                        # pandas doesn't have ignore separator feature
    feat_df           = pd.read_csv(ff, names='x', sep='☺️', engine='python')
    label_df          = pd.read_csv(lf, names='y')
    rater_df          = pd.concat([feat_df, label_df], axis=1)
    rater_df['rater'] = r
    cornell_df        = pd.concat([cornell_df, rater_df], axis=0, ignore_index=True)


## Numerical labels for raters (e.g. Dennis -> 0, James -> 1, ect.)
_, cornell_df['rater_id'] = np.unique(cornell_df['rater'], return_inverse=True)

cornell_dict = {}

## Tokenize
cornell_dict['x_tok']    = tokenize(cornell_df['x'])

## Padding
cornell_dict['x_padded'] = add_padding(cornell_dict['x_tok'], dtype=np.dtype('<U32'))

## Embedding
cornell_dict['x_embed']  = embed_transform(cornell_dict['x_padded'], size=20)

## Save data
with pd.HDFStore(PANDAS) as store:
    store.append('cornell_df', cornell_df)

with open(PICKLE, 'wb') as f:
    pickle.dump(cornell_dict, f)
