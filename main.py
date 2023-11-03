import pandas as pd
from data_tokenization import preprocess_text
from data_preprocessing import preprocess_data
from sentiment_analysis import perform_sentiment_analysis

if __name__ == "__main__":

    # Clean up the dataset
    preprocess_data('Data/reviews.csv')

    # Read the text data from the csv file
    text_data = pd.read_csv(
        'Data/reviews_edited.csv')['Review Text'].values.tolist()

    review_rating = pd.read_csv(
        'Data/reviews_edited.csv')['Rating'].values.tolist()

    # Turn the unstructured text data into structured clean text
    processed_text_data = preprocess_text(text_data)

    # Perform Sentiment Analysis and Caluclate performance metrics
    accuracy, report = perform_sentiment_analysis(
        processed_text_data, review_rating)

    # Print the results
    print("Accuracy:", accuracy)
    print(report)
