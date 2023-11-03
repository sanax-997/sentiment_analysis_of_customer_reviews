import pandas as pd


def preprocess_data(file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Remove rows with empty "Review Text" entries
    df = df.dropna(subset=['Review Text'])

    # Save the dataframe
    df.to_csv('Data/reviews_edited.csv')
