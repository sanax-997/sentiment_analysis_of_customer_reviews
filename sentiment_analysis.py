from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def perform_sentiment_analysis(processed_text_data, review_rating):
    # Initialize a BERT based model
    sentiment_pipeline = pipeline(
        model='DataMonke/bert-base-uncased-finetuned-review-sentiment-analysis')

    # Combine the tokenized text to a list of strings
    sentence_list = [' '.join(doc) for doc in processed_text_data]

    # Make predictions with the sentiment model
    predictions = sentiment_pipeline(sentence_list)

    # Extract and map the predicted labels
    predicted_labels = [result['label'] for result in predictions]

    # Map the rating of the predictions to sentiment categories
    predicted_labels_converted = []
    for rating in predicted_labels:
        rating = int(rating[0])
        if rating == 1 or rating == 2:
            predicted_labels_converted.append("negative")
        elif rating == 3:
            predicted_labels_converted.append("neutral")
        elif rating == 4 or rating == 5:
            predicted_labels_converted.append("positive")

    # Create a visualization of the classification results
    # Count the number of positive, neutral, and negative entries
    sentiment_counts = {
        "positive": predicted_labels_converted.count("positive"),
        "neutral": predicted_labels_converted.count("neutral"),
        "negative": predicted_labels_converted.count("negative"),
    }

    # Define colors for the bars
    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}

    # Create a bar plot
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=[colors[sentiment] for sentiment in sentiment_counts.keys()])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    
    # Save the plot as an image file
    plt.savefig('sentiment_distribution.png')

    # Map the rating of the data to sentiment categories
    sentiment_labels = []
    for rating in review_rating:
        if rating in [1, 2]:
            sentiment_labels.append("negative")
        elif rating == 3:
            sentiment_labels.append("neutral")
        elif rating in [4, 5]:
            sentiment_labels.append("positive")

    # Calculate accuracy
    accuracy = accuracy_score(sentiment_labels, predicted_labels_converted)

    # Generate a classification report
    report = classification_report(sentiment_labels, predicted_labels_converted, target_names=[
                                   "negative", "neutral", "positive"])

    return accuracy, report
