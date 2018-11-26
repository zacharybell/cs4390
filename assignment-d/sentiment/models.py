import pandas as pd
import pickle


from preprocessing import PICKLE, PANDAS

## Read data
with open(PICKLE, 'rb') as f:
    cornell_dict = pickle.load(f)

with pd.HDFStore(PANDAS) as store:
    cornell_df = store.get('cornell_df') 

print(cornell_df.head())