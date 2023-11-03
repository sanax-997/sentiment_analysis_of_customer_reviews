from transformers import AutoTokenizer


def preprocess_text(text_data):
    # Initialize a tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(
        'DataMonke/bert-base-uncased-finetuned-review-sentiment-analysis')

    # Initalize an empty list for the text data
    document_list = []

    # Iterate through the entire text corpus
    for text in text_data:
        # Tokenize the text
        tokens = tokenizer.tokenize(text)

        # Add tokens to the document list
        document_list.append(tokens)

    return document_list
