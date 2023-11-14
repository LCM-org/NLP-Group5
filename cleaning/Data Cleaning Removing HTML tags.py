
#Kanika Kataria - c0866652
#Data Cleaning - Removing HTML Tags

import re
import pandas as pd

path = "C:\\Users\\katar\\OneDrive\\Desktop\\Sem 3 - Imp\\NLP - Wed\\Project\\measuring_hate_speech.csv"
#reading dataset
df = pd.read_csv(path)

#defining a function to remove html tags from text string
def remove_html_tags(text):
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
    print(text)

#column text with HTML tags in dataset
text_df = 'text'

# Applying the remove_html_tags function to text column
df[text_df] = df[text_df].apply(remove_html_tags)

df.head()

df[[text_df]].head(100)
