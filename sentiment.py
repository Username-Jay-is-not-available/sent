# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions
import string
import re
import classifier

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
    vocab = sorted(set(preprocessed_text))  
    return vocab


def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """
    vectorized_text = []
    for str in vocab:
        if str in text:
            vectorized_text.append('1')
        else:
            vectorized_text.append('0')


    label = text[-1]
    return vectorized_text, label


def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """
    score = 0

    for p, t in zip(predicted_labels, true_labels):
        if p == t:
            score += 1
        
    total = len(predicted_labels)
    accuracy_score = score / total
    return accuracy_score


def main():
    # Take in text files and outputs sentiment scores

    return 1


if __name__ == "__main__":
    main()