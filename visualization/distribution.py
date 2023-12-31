import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Load  CSV file
df = pd.read_csv('measuring_hate_speech.csv')


text_data = " ".join(df['text'].astype(str))

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Text Data Distribution')
plt.show()

# Choose the sentiment column
sentiment_column = 'sentiment'

# Visualize sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=sentiment_column, data=df, palette='viridis')
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Choose the hate speech score column
hate_speech_score_column = 'hate_speech_score'

# Visualize the distribution of hate speech scores
plt.figure(figsize=(10, 6))
sns.histplot(df[hate_speech_score_column], kde=True, color='darkblue')
plt.title('Distribution of Hate Speech Scores')
plt.xlabel('Hate Speech Score')
plt.ylabel('Frequency')
plt.show()
