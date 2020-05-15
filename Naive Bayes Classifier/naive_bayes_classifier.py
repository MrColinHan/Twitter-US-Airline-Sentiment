"""
@author: Changze Han
"""

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import random
import csv
import re
import string
from nltk import classify
from nltk import NaiveBayesClassifier
from sklearn.metrics import f1_score
import collections
import pandas
import matplotlib.pyplot as plt
import numpy as np
import itertools

'''
    This program performs four tasks: 
        1. Tokenize text and text stemming
        2. Train and Testing the Naive Bayes model
	    3. visualize confusion matrix
	    4. calculate precision, recall, f-measure
    
'''
temp_matrix = None
# =============================================================================
positive_input_file_dir = r"/Users/Han/Downloads/web project data/positive_tweets.csv"
negative_input_file_dir = r"/Users/Han/Downloads/web project data/negative_tweets.csv"
train_text_column_index = 10  # col index starts from 0

# file that will be used for prediction
predict_input_file_dir = r"/Users/Han/Downloads/web project data/parsed_tweet_0.csv"
predict_text_column_index = 3

output_file_dir = r"/Users/Han/Downloads/web project data/out.csv"

# =============================================================================
positive_tokens = []
cleaned_positive_tokens = []
negative_tokens = []
cleaned_negative_tokens = []

predict_tokens = []
cleaned_predict_tokens = []

output_list = []


def read_csv(filedir, listname):
    file = open(filedir)
    reader = csv.reader(file)
    for row in reader:
        listname.append(row)


def write_csv(x, y):
    with open(y,'w+') as file:
        wr = csv.writer(file, dialect='excel')
        wr.writerows(x)
    file.close()


# file_dir is the input csv file directory
# txt_col_i is the column index of tweet texts
# token_list is an empty list that will be used to save tokens
# cleaned_token_list is another empty list what will be used to save cleaned tokens

def clean_up_tweets(file_dir, txt_col_i, token_list, cleaned_token_list):
    input_list = []
    tweets_list = []
    tweets_list_tokenized = []
    tweets_list_tokenized_tagged = []
    tweets_list_tokenized_tagged_lemma = []

    # read tweet texts:
    read_csv(file_dir, input_list)
    for row in input_list[1:]:  # exclude header row
        tweets_list.append(row[txt_col_i])

    # tokenize texts:
    t_tkn = TweetTokenizer()
    for tweet in tweets_list:
        tweets_list_tokenized.append(t_tkn.tokenize(tweet))
        token_list.append(t_tkn.tokenize(tweet))

    # add tags:
    # NN: noun;  VB: verb
    for tweet_tokenized in tweets_list_tokenized:
        tweets_list_tokenized_tagged.append(pos_tag(tweet_tokenized))

    # Lemmatize text:
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    for tweet_tokenized_tagged in tweets_list_tokenized_tagged:
        current_text_lemma = []
    # remove stop words, hyperlinks, Twitter handles, Punctuation and special characters:
        for token, tag in tweet_tokenized_tagged:
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            token = lemmatizer.lemmatize(token, pos)
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                current_text_lemma.append(token.lower())
        #tweets_list_tokenized_tagged_lemma.append(current_text_lemma)
        cleaned_token_list.append(current_text_lemma)

    #print(token_list[4])
    #print(cleaned_token_list[4])


# Converting Tokens to a Dictionary: dictionary with words as keys and True as values
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def main():
    global positive_tokens
    global cleaned_positive_tokens
    global negative_tokens
    global cleaned_negative_tokens
    global predict_tokens
    global cleaned_predict_tokens
    global output_list

    global temp_matrix

    # get cleaned up tokens
    print("......Cleaning up Dataset......")
    print("...tokenizing...")
    print("...normalizing...")
    print("...Lemmatizing...")
    print("...removing stop words...\n")
    clean_up_tweets(positive_input_file_dir, train_text_column_index, positive_tokens, cleaned_positive_tokens)
    print("Done: clean up positive tweets")
    clean_up_tweets(negative_input_file_dir, train_text_column_index, negative_tokens, cleaned_negative_tokens)
    print("Done: clean up negative tweets\n")

    #print(positive_tokens[4])
    #print(cleaned_positive_tokens[4])
    #print(negative_tokens[4])
    #print(cleaned_negative_tokens[4])

    # Converting Tokens to a Dictionary:
    positive_tokens_for_model = get_tweets_for_model(cleaned_positive_tokens)
    negative_tokens_for_model = get_tweets_for_model(cleaned_negative_tokens)
    print("Done: Convert tokens to dictionaries.\n")

    # create a dataset by joining the positive and negative tweets.
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]
    dataset = positive_dataset + negative_dataset
    print("Done: Combine dataset by joining the positive and negative tweets.")
    # random shuffle
    random.shuffle(dataset)

    print(f"positive dataset: {len(positive_dataset)} tweets.")
    print(f"negative dataset: {len(negative_dataset)} tweets.")
    print(f"combine positive & negative dataset: {len(dataset)} tweets.\n")
    print("......Training Data......")

    # splits the shuffled data into a ratio of 7:3 for training and testing
    train_data = dataset[:round(len(dataset)*0.7)]
    test_data = dataset[round(len(dataset)*0.7):]
    print(f"train data: {len(train_data)} tweets")
    print(f"test data: {len(test_data)} tweets\n")

    print("Build & Test Naive_Bayes_Classifier Model: ")
    classifier = NaiveBayesClassifier.train(train_data)
    print("=============Accuracy====================")
    print(f"Accuracy is:{classify.accuracy(classifier, test_data)}\n")

    print(classifier.show_most_informative_features(10))

    # build confusion matrix
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    labels = []
    tests = []

    for i, (feats, label) in enumerate(test_data):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
        labels.append(label)
        tests.append(observed)
    print("=============Precision and Recall====================")
    print(f"Positive precision: {nltk.precision(refsets['Positive'], testsets['Positive'])}")
    print(f"Positive recall: {nltk.recall(refsets['Positive'], testsets['Positive'])}")
    print(f"Positive F-measure: {nltk.f_measure(refsets['Positive'], testsets['Positive'])}")
    print(f"Negative precision: {nltk.precision(refsets['Negative'], testsets['Negative'])}")
    print(f"Negative recall: {nltk.recall(refsets['Negative'], testsets['Negative'])}")
    print(f"Negative F-measure: {nltk.f_measure(refsets['Negative'], testsets['Negative'])}")

    print("=============Confusion Matrix====================")
    confusion_matrix_result = nltk.ConfusionMatrix(labels, tests)
    print(confusion_matrix_result)

    # now visualize the confusion matrix using matplotlib.pyplot
    #=============Visualize Confusion Matrix====================
    # matirx needs to be saved as np.array()
    # also, needs to extract ._confusion first
    confusion_matrix_result = np.array(confusion_matrix_result._confusion)
    temp_matrix = confusion_matrix_result

    classes = ["Negatives", "Positives"]
    plt.figure()
    plt.imshow(confusion_matrix_result, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    text_format = 'd'
    thresh = confusion_matrix_result.max()/2
    for row, column in itertools.product(range(confusion_matrix_result.shape[0]),
                                         range(confusion_matrix_result.shape[1])):
        plt.text(column, row, format(confusion_matrix_result[row, column], text_format),
                 horizontalalignment='center',
                 color='white' if confusion_matrix_result[row, column] > thresh else "black")

    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    plt.tight_layout()
    # needs a high resolution image
    plt.savefig("/Users/Han/Downloads/web project data/confusion_matrix.png", dpi=1200)
    plt.show()

    # =======================================now predict new tweets=======================================
    print("......Now Cleaning up new Dataset......")
    print("...tokenizing...")
    print("...normalizing...")
    print("...Lemmatizing...")
    print("...removing stop words...\n")
    clean_up_tweets(predict_input_file_dir, predict_text_column_index, predict_tokens, cleaned_predict_tokens)
    print("Done: clean up predict tweets\n")

    print("...Now Deploy Bayes Classifier on new dataset...")
    for current_tweet_tokens in cleaned_predict_tokens:
        output_list.append([classifier.classify(dict([token, True] for token in current_tweet_tokens))])

    write_csv(output_list, output_file_dir)
    print("Done! ")


main()











