
import pandas as pd
import emoji

def remove_emojis(df):
    # Apply the emoji.demojize function to the 'text' column
    df['text'] = df['text'].apply(lambda x: emoji.demojize(x))
    return df

# Load CSV data into a DataFrame
df = pd.read_csv('measuring_hate_speech.csv')

# Call the remove_emojis function to modify the DataFrame
cleaned_df = remove_emojis(df)

# Display the modified 'text' column
print(cleaned_df['text'])

