import matplotlib.pyplot as plt
import pandas as pd

def remove_boxplot_outliers(df):

    # Select only numerical columns
    numerical_columns = df.select_dtypes(include=['number'])
    numerical_columns = numerical_columns.drop(columns=['comment_id', 'annotator_id'])

    # Highlight outliers
    for col in numerical_columns.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_no_outliers = df[(df[col] >= lower_bound) | (df[col] <= upper_bound)]

    return df_no_outliers


#for testing
df = pd.read_csv('measuring_hate_speech.csv')

#Before Count of Dataframe
print(df.shape[0])

#After applying the function to remove a boxplot outliers
print(remove_boxplot_outliers(df).shape[0])