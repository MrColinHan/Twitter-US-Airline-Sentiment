import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Read in tweets dataset
tweets = pd.read_csv("combinedTweets.csv")

# Use text to predict airline_sentiment_gold (NLTK)
# Generate word cloud for positive and negative words using text
# 

#--------------------------------------------------------------------------------------------
# Question: Which airlines get tweeted about the most?
# Bar graph with numbers
# Pie chart with frequencies
# Data Frame
#--------------------------------------------------------------------------------------------
numberOfTweetsPerAirline = tweets.airline.value_counts()
print("Number of Tweets Per Airline (airline)")
print(numberOfTweetsPerAirline)
print()

# Using Pie chart to plot frequency of tweets per airline
words = [('United', 3822), ('US Airways', 2913), ('American', 2759), ('Southwest', 2420), ('Delta', 2222), ('Virgin America', 504)]
sizes, labels = [i[1] for i in words],[i[0] for i in words]
plt.title("Frequency of Tweets Per Airline")
plt.pie(sizes, labels=labels,autopct='%1.1i%%')
plt.show()

#--------------------------------------------------------------------------------------------
# Question: What percentage of all the tweets are positive, negative and neutral?
# Bar graph with numbers
# Data Frame 
#--------------------------------------------------------------------------------------------
numberOfTweetsBySentiment = tweets.airline_sentiment.value_counts()
print("Total Number of Negative, Neutral and Positive Tweets")
print(numberOfTweetsBySentiment)
print()
names = list(numberOfTweetsBySentiment.keys())
plt.title('Number of Negative, Neutral and Positive Tweets')
plt.xlabel("Mood")
plt.ylabel("Number of Tweets")
plt.bar(names, numberOfTweetsBySentiment, color =('red', 'blue', 'green'))
plt.show()

"""
#--------------------------------------------------------------------------------------------
# Question: What are the top reasons for poor service?
#--------------------------------------------------------------------------------------------
print("Top Ten Reasons for Poor Service (negativereason)")
topTenNegativeReasons = tweets.negativereason.value_counts()
print(topTenNegativeReasons)
print()
reasons = list(topTenNegativeReasons.keys())
reasons = [x.replace(' ', '\n') for x in reasons]
plt.title('Top Ten Reasons for Poor Service')
plt.xlabel("Negative Reason")
plt.ylabel("Number of Tweets")
colors = ("red", "blue", "green", "black", "brown", "yellow", "orange", "purple", "gray", "pink")
plt.bar(reasons, topTenNegativeReasons, color=colors)
plt.show()
"""

"""
#--------------------------------------------------------------------------------------------
# Best Airlines by Number of Positive Reasons (airline_sentiment)
# Pie chart with frequencies
# Data Frame
#--------------------------------------------------------------------------------------------
print("Best Airlines by Number of Positive Reasons (airline_sentiment)")
numberOfTweetsBySentiment = tweets.airline_sentiment.value_counts()
is_positive = tweets.airline_sentiment.str.contains("positive")
positive_tweets = tweets[is_positive]
positive_tweets.shape
best_airline = positive_tweets[['airline','airline_sentiment_confidence']]
#best_airline = (best_airline.groupby('airline').airline_sentiment_confidence.count()/len(best_airline)).sort_values(ascending=False)
best_airline = best_airline.groupby('airline').airline_sentiment_confidence.count().sort_values(ascending=False)
print(best_airline)
print()
"""
# Pie chart Best Airlines by Number of Positive Reasons
words = [('Southwest', 570), ('Delta', 544), ('United', 492), ('American', 336), ('US Airways', 269), ('Virgin America', 152)]
sizes, labels = [i[1] for i in words],[i[0] for i in words]
plt.title("Best Airlines by Number of Positive Reasons (airline_sentiment)")
plt.pie(sizes, labels=labels,autopct='%1.1i%%')
plt.show()

"""
#--------------------------------------------------------------------------------------------
# Worst Airlines by Number of Negative Reasons (negativereason)
# Pie chart with frequencies
# Data Frame
#--------------------------------------------------------------------------------------------
print("Worst Airlines by Number of Negative Reasons (negativereason)")
is_negative = tweets.airline_sentiment.str.contains("negative")
negative_tweets = tweets[is_negative]
negative_tweets.shape
worst_airline = negative_tweets[['airline','negativereason']]
worst_airline = worst_airline.groupby('airline').negativereason.count().sort_values(ascending=False)
print(worst_airline)

# Pie chart for Worst Airlines by Number of Negative Reasons (negativereason)
words = [('United', 2633), ('US Airways', 2263), ('American', 1960), ('Southwest', 1186), ('Delta', 955), ('Virgin America', 181)]
sizes, labels = [i[1] for i in words],[i[0] for i in words]
plt.title("Worst Airlines by Number of Negative Reasons (negativereason)")
plt.pie(sizes, labels=labels,autopct='%1.1i%%')
plt.show()
"""




# Total number of positive, negative and neutral tweets by Airline
sentimentByAirline = pd.crosstab(tweets.airline, tweets.airline_sentiment).apply(lambda x: x/x.sum() * 100, axis=1)
sentimentByAirline.plot.bar(stacked=True, figsize=(6,6), title='Frequency of Negative, Neutral and Positive Tweets By Airline')
plt.legend(bbox_to_anchor=(1.1, 1))
plt.xlabel('Airline')
plt.ylabel('Number of Tweets')
plt.show()

"""
# Top Ten Reasons for Poor Service By Airline
reasonsByAirline = pd.crosstab(tweets.airline, tweets.negativereason).apply(lambda x: x/x.sum() * 100, axis=1)
reasonsByAirline.plot.bar(stacked=True, figsize=(6,6), title='Frequency of Top Ten Negative Reasons By Airline')
plt.xlabel('Airline')
plt.ylabel('Top Ten Negative Reasons')
plt.legend(bbox_to_anchor=(0.95, 0.7))
plt.show()
"""

######### Machine Learning part for airline_sentiment_gold #############
analyser = SentimentIntensityAnalyzer()

# Columns needed for classification 
# tweets = tweets[['airline_sentiment', 'airline','text' ]]
tweets = tweets[['tweet_id', 'airline_sentiment', 'airline', 'name', 'text', 'tweet_coord', 'tweet_created', 'tweet_location']]

# Print number of tweets
print()
print("Number of Tweets is: " + str(len(tweets)))

########## Compute VADER scores ###################

i=0 #counter
compval1 = []  #empty list to hold our computed 'compound' VADER scores
while (i<len(tweets)):
    k = analyser.polarity_scores(tweets.iloc[i]['text'])
    compval1.append(k['compound'])
    i = i+1
    
#converting sentiment values to numpy for easier usage
compval1 = np.array(compval1)
#len(compval1)

# Add new column called VADER score with computed values
tweets['VADER Score'] = compval1
#tweets.to_csv("newTweets.csv", index=False)
outfile = open('newTweets.csv', 'w')
tweets.to_csv(outfile, index=False)

####### Using VADER Scores for airline_sentiment_gold column ########
i = 0

#empty series to hold our predicted values
predicted_value = [] 

# Use VADER scores to determine sentiment
while(i<len(tweets)):
    if ((tweets.iloc[i]['VADER Score'] >= 0.7)):
        predicted_value.append('positive')
        i = i+1
    elif ((tweets.iloc[i]['VADER Score'] > 0) & (tweets.iloc[i]['VADER Score'] < 0.7)):
        predicted_value.append('neutral')
        i = i+1
    elif ((tweets.iloc[i]['VADER Score'] <= 0)):
        predicted_value.append('negative')
        i = i+1

# Add new column called airline_sentiment_gold using VADER score with computed values
tweets['vader_airline_sentiment'] = predicted_value
#tweets.to_csv("newTweets.csv", index=False)
outfile = open('combinedTweetsFile.csv', 'w')
tweets.to_csv(outfile, index=False)
outfile.close()

################### Prediction Accuracy, comparing difference between airline_sentiment_gold
# and airline_sentiment

correctPredictions = tweets[tweets['airline_sentiment'] == tweets['vader_airline_sentiment']]
print("Percentage of correct predictions: " + str(len(correctPredictions)/len(tweets)))
#exit()
