#importing the necessary library...

import pandas as pd


#reading the csv file....
df = pd.read_csv("C:\\Users\\virkn\\Downloads\\NLP\\measuring_hate_speech.csv",encoding='utf-8')

#displaying the first five records...
df.head()


#checking column names before applying contractions...
print("Column names:", df.columns)


#printing the result of original column named text....
original_text_column = df['text']
print("Original 'text' column:")
print(original_text_column.head())


import contractions
#defining a function to expand contraction...
def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text


#applying the expanding contraction to the text column...
df['text'] = df['text'].apply(lambda x: expand_contractions(x) if isinstance(x, str) else x

#printing the result of expanded text column...
expanded_text_column = df['text']
print("\nText column after expanding contractions:")
print(expanded_text_column.head())


#saving the expanded data to the new csv file...
df.to_csv('expanded_hate_speech_dataset.csv', index=False, encoding='utf-8')


#reading the expanded csv file and displaying the results...
df=pd.read_csv("expanded_hate_speech_dataset.csv",encoding='utf-8')
df.head(4)


# Display original and expanded text for the first few rows to understand the difference...
for index in range(10):
    original_text = original_text_column.iloc[index]
    expanded_text = expanded_text_column.iloc[index]

    print(f"\nExample {index + 1}:\n")
    print(f"Original Text: \n{original_text}\n")
    print(f"Expanded Text: \n{expanded_text}\n")


# As we can see in the example 9, the text they'll is converted to they will after the expanding contraction...
