import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 


def plot_heatmap(df):
    numerical_df = df.select_dtypes(include='number')
    numerical_df = numerical_df.drop(columns=['comment_id', 'annotator_id'])
    correlation_matrix = numerical_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    
    plt.title('Correlation Heatmap for Numerical Columns')
    plt.show()

    # Return the correlation matrix
    return correlation_matrix

# Reading csv file
df = pd.read_csv(r"C:\Users\Owner\OneDrive\Documents\measuring_hate_speech.csv")

# Plot the heatmap for numerical columns and get the correlation matrix
correlation_matrix = plot_heatmap(df)

