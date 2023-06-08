# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions
import string
import re
import numpy as np
from collections import Counter
from classifier import BayesClassifier

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
    vectorized_text = []
    labels = []
    
    # Split the text into sentences
    sentences = text.split('\n')
    
    for sentence in sentences:
        # Skip empty lines
        if not sentence.strip():
            continue

        # Split the sentence into words and label
        try:
            words, label = sentence.split('\t')
        except ValueError:
            print(f"Skipping malformed sentence: {sentence}")
            continue

        # Convert the sentence to a vector
        vector = [word for word in process_text(words) if word in vocab]
        
        vectorized_text.append(vector)
        labels.append(int(label))
    
    return vectorized_text, labels


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

    with open('trainingSet.txt', 'r') as f:
        training_data = f.read()

    with open('testSet.txt', 'r') as f:
        test_data = f.read()

    preprocessed_training_data = process_text(training_data)

    vocab = build_vocab(preprocessed_training_data)

    vectorized_training_data, training_labels = vectorize_text(training_data, vocab)
    vectorized_test_data, test_labels = vectorize_text(test_data, vocab)

    classifier = BayesClassifier()
    classifier.train(vectorized_training_data, training_labels, vocab)

    predicted_labels = classifier.classify_text(vectorized_test_data, vocab)
    
    # https://www.askpython.com/python/built-in-methods/python-print-to-file
    print("Accuracy: ", accuracy(predicted_labels, test_labels), file=open('results.txt', 'a'))
    

    return 1


if __name__ == "__main__":
    main()
