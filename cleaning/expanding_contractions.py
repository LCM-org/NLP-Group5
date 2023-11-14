# Expanding Contractions 


# Importing necessary libraries 

import pandas as pd
import contractions


#reading csv file.......
df = pd.read_csv("measuring_hate_speech.csv")

#defining contraction function...
def expand_contractions(df):
  df= df['text'].apply(contractions.fix)
  return df

# Display the original and expanded text side by side
print(df['text'])

expand_df = expand_contractions(df)
print(expand_df['text'])




