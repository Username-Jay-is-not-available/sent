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
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.replace("'", '')  # Remove apostrophes
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
    preprocessed_testing_data = process_text(test_data)

    vocab = build_vocab(preprocessed_training_data)

    vectorized_training_data, training_labels = vectorize_text(training_data, vocab)
    vectorized_test_data, test_labels = vectorize_text(test_data, vocab)
    
    # Writing preprocessed training data
    with open("preprocessed_train.txt", "w") as f:
        # Write the vocab
        for word in vocab:
            f.write(word + ',')
        f.write('classlabel\n')

        # Write the vectors and labels
        for i in range(len(vectorized_training_data)):
            vector = vectorized_training_data[i]
            label = training_labels[i]
            
            # Write the vector
            for word in vocab:
                if word in vector:
                    f.write('1,')
                else:
                    f.write('0,')
            
            # Write the label
            f.write(str(label) + '\n')

    # Writing preprocessed testing data
    with open("preprocessed_test.txt", "w") as f:
        for word in vocab:
            f.write(word + ',')
        f.write('classlabel\n')

        for i in range(len(vectorized_test_data)):
            vector = vectorized_test_data[i]
            label = test_labels[i]
            
            for word in vocab:
                if word in vector:
                    f.write('1,')
                else:
                    f.write('0,')
            
            f.write(str(label) + '\n')

    
    classifier = BayesClassifier()
        
    for i, section in enumerate(classifier.file_sections):
        # Use the section as an index for incremental training
        incremental_data = vectorized_training_data[:section]
        incremental_labels = training_labels[:section]

        classifier.train(incremental_data, incremental_labels, vocab)

        predicted_training_labels = classifier.classify_text(vectorized_training_data, vocab)
        
        predicted_testing_labels = classifier.classify_text(vectorized_test_data, vocab)
        
        print(f"line 1-{section} used for training to test trainingSet.txt with the accuracy of: ", accuracy(predicted_training_labels, training_labels), file=open('results.txt', 'a'))

        print(f"line 1-{section} used for training to test testSet.txt with the accuracy of: ", accuracy(predicted_testing_labels, test_labels), file=open('results.txt', 'a'))

    return 1


if __name__ == "__main__":
    main()