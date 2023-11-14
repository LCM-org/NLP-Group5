import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 


#Reading csv file 
df = pd.read_csv('measuring_hate_speech.csv')


numerical_df = df.select_dtypes(include='number')

def plot_heatmap(df):
    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)

    # Display the plot
    plt.title('Correlation Heatmap for Numerical Columns')
    plt.show()

    # Return the correlation matrix
    return correlation_matrix

# Plot the heatmap for numerical columns and get the correlation matrix
correlation_matrix = plot_heatmap(numerical_df)

