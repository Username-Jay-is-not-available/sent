# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions
import string
import re

def process_text(text):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """
    text = text.lower()  # Convert to lower case
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    preprocessed_text = text.split()  # Split into words
    return preprocessed_text


def build_vocab(preprocessed_text):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """
    #removed duplicated words from preprocessed_text and sorted it
    vocab = sorted(set(preprocessed_text))
    return vocab


def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """
    return vectorized_text, labels


def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """

    return accuracy_score


def main():
    # Take in text files and outputs sentiment scores

    return 1


if __name__ == "__main__":
    main()