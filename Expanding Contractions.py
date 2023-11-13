# Expanding Contractions 


# importing necessary libraries 

import pandas as pd
import contractions

#Reading csv file 
df = pd.read_csv(r"C:\Users\Owner\OneDrive\Documents\measuring_hate_speech.csv")
print(df.head())


def expand_contractions(text):
   expanded_text = contractions.fix(text)
   return expanded_text

# Apply expand contraction function to text column 
df['expanded_text'] = df['text'].apply(expand_contractions)

# Display the original and expanded text side by side
print(df[['text', 'expanded_text']])




