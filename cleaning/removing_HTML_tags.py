
#Kanika Kataria - c0866652
#Data Cleaning - Removing HTML Tags

import re
import pandas as pd

path = "measuring_hate_speech.csv"
#reading dataset
df = pd.read_csv(path)

def remove_html_tags(df):
    def remove_html_tags(text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    df['text'] = df['text'].apply(remove_html_tags)
    return df


print(df['text'].head(20))

df_cleaned = remove_html_tags(df)
print(df_cleaned['text'].head(20))


