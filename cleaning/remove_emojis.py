
import pandas as pd
import emoji

def remove_emojis(text):
    return emoji.demojize(text)

# Load CSV data into a DataFrame
df = pd.read_csv('measuring_hate_speech.csv')

# Apply the remove_emojis function to the relevant column(s)
df['text'] = df['text'].apply(remove_emojis)

print(df['text'])

