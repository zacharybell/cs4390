#%% 
import pandas as pd

PANDAS = 'data/pandas.pkl'

cornell_df = pd.read_pickle(PANDAS)

import gensim
from gensim.models import Word2Vec

#%%
w2v = gensim.models.Word2Vec(sentences=cornell_df['x_tok'], min_count=2, size=20)

def sent_to_vec(model: Word2Vec, sent: list):
    import numpy as np
    return np.array([model.wv[w] for w in sent])


#%%
import math
import numpy as np

cornell_docs_padded = cornell_df['x_tok']
doc_lengths         = [ len(d) for d in cornell_docs_padded ]
doc_avg_len         = np.mean(doc_lengths, dtype=int)

for i, d in enumerate(cornell_docs_padded):
    d_len = len(d)
    
    if (d_len > doc_avg_len):
        cornell_docs_padded[i] = d[:doc_avg_len]
    else:
        cornell_docs_padded[i] = np.pad(d, (0, doc_avg_len - d_len), 'constant', constant_values=('', ''))

print(np.max(cornell_docs_padded))
print(np.min(cornell_docs_padded))