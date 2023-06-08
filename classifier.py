# This file implements a Naive Bayes Classifier
import math

class BayesClassifier():
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """
    def __init__(self):
        self.postive_word_counts = {}
        self.negative_word_counts = {}
        self.percent_positive_scentences = 0
        self.percent_negative_scentences = 0
        self.file_length = 499
        self.file_sections = [self.file_length // 4, self.file_length // 2, (3 * self.file_length) // 4, self.file_length]


    def train(self, train_data, train_labels, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """
        #init variables for positive and negative words AND negative and negative sentences
        total_positive_words = 0
        total_negative_words = 0
        total_positive_sentences = 0
        total_negative_sentences = 0

        #loop through training data
        for sentence, label in zip(train_data, train_labels):
            #if positive (1)
            if label == 1:
                total_positive_sentences += 1
                #add positive words into list
                for word in sentence:
                    if word not in self.postive_word_counts:
                        self.postive_word_counts[word] = 0
                    self.postive_word_counts[word] += 1
                    total_positive_words += 1
            #if negative (0)
            else:
                total_negative_sentences += 1
                #add negative words into list
                for word in sentence:
                    if word not in self.negative_word_counts:
                        self.negative_word_counts[word] = 0
                    self.negative_word_counts[word] += 1
                    total_negative_words += 1

        #calculate ratio of positive and negative sentences
        self.percent_positive_scentences = total_positive_sentences / (total_positive_sentences + total_negative_sentences)
        self.percent_negative_scentences = total_negative_sentences / (total_positive_sentences + total_negative_sentences)

        #error checking: make sure every word is entried in either positive word list or negative word list
        for word in vocab:
            if word not in self.postive_word_counts:
                self.postive_word_counts[word] = 0
            if word not in self.negative_word_counts:
                self.negative_word_counts[word] = 0

        #store data
        self.total_positive_words = total_positive_words
        self.total_negative_words = total_negative_words
        self.vocabulary_length = len(vocab)

    def classify_text(self, vectors, vocab):
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """
        predictions = []

        #loop through each sentence
        for sentence in vectors:
            #log probability of a sentence being positive
            log_positive_probability = math.log(self.percent_positive_scentences)
            #log probability of a sentence being negative
            log_negative_probability = math.log(self.percent_negative_scentences)

            for word in sentence:
                #add to the total positive and negative log probability
                word_positive_probability = (self.postive_word_counts[word] + 1) / (self.total_positive_words + self.vocabulary_length)
                word_negative_probability = (self.negative_word_counts[word] + 1) / (self.total_negative_words + self.vocabulary_length)

                log_positive_probability += math.log(word_positive_probability)
                log_negative_probability += math.log(word_negative_probability)

            #predict wether the sentence is positive or negative
            if log_positive_probability > log_negative_probability:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions
    