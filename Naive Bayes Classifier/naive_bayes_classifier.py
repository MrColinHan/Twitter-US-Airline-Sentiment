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

'''
    This program performs two tasks: 
        1. Tokenize text and text stemming
        2. Train ans Testing the model
    
'''

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

    print(f"negative dataset: {len(positive_dataset)} tweets.")
    print(f"positive dataset: {len(negative_dataset)} tweets.")
    print(f"combine positive & negative dataset: {len(dataset)} tweets.\n")
    print("......Training Data......")

    # splits the shuffled data into a ratio of 7:3 for training and testing
    train_data = dataset[:round(len(dataset)*0.7)]
    test_data = dataset[round(len(dataset)*0.7):]
    print(f"train data: {len(train_data)} tweets")
    print(f"test data: {len(test_data)} tweets\n")

    print("Build & Test Naive_Bayes_Classifier Model: ")
    classifier = NaiveBayesClassifier.train(train_data)

    print(f"Accuracy is:{classify.accuracy(classifier, test_data)}\n")

    print(classifier.show_most_informative_features(10))

    # =======================================now predict new tweets=======================================
    print("......Now Cleaning up new Dataset......")
    print("...tokenizing...")
    print("...normalizing...")
    print("...Lemmatizing...")
    print("...removing stop words...\n")
    clean_up_tweets(predict_input_file_dir, predict_text_column_index, predict_tokens, cleaned_predict_tokens)
    print("Done: clean up predict tweets\n")

    print("...Now Deploy Bayes Classifier...")
    for current_tweet_tokens in cleaned_predict_tokens:
        output_list.append([classifier.classify(dict([token, True] for token in current_tweet_tokens))])

    write_csv(output_list, output_file_dir)
    print("Done! ")


main()











