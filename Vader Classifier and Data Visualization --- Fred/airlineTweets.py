import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Read in tweets dataset
tweets = pd.read_csv("finalInput.csv")

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
words = [('United', 4648), ('US Airways', 2913), ('American', 3450), ('Southwest', 2814), ('Delta', 3052), ('Virgin America', 568), ('JetBlue', 238)]
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
plt.xticks(fontsize=7)
plt.show()

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
words = [('United', 3301), ('US Airways', 2263), ('American', 2518), ('Southwest', 1440), ('Delta', 1588), ('Virgin America', 230), ('JetBlue', 163)]
sizes, labels = [i[1] for i in words],[i[0] for i in words]
plt.title("Worst Airlines by Number of Negative Reasons (negativereason)")
plt.pie(sizes, labels=labels,autopct='%1.1i%%')
plt.show()





# Total number of positive, negative and neutral tweets by Airline
sentimentByAirline = pd.crosstab(tweets.airline, tweets.airline_sentiment).apply(lambda x: x/x.sum() * 100, axis=1)
sentimentByAirline.plot.bar(stacked=True, figsize=(6,6), title='Frequency of Negative, Neutral and Positive Tweets By Airline')
plt.legend(bbox_to_anchor=(1.1, 1))
plt.xlabel('Airline')
plt.ylabel('Number of Tweets')
plt.show()


# Top Ten Reasons for Poor Service By Airline
reasonsByAirline = pd.crosstab(tweets.airline, tweets.negativereason).apply(lambda x: x/x.sum() * 100, axis=1)
reasonsByAirline.plot.bar(stacked=True, figsize=(6,6), title='Frequency of Top Ten Negative Reasons By Airline')
plt.xlabel('Airline')
plt.ylabel('Top Ten Negative Reasons')
plt.legend(bbox_to_anchor=(0.95, 0.7))
plt.show()


######### Machine Learning part for sentiment #############
analyser = SentimentIntensityAnalyzer()

# Columns needed for classification 
# tweets = tweets[['airline_sentiment', 'airline','text' ]]
tweets = tweets[['tweet_id', 'airline_sentiment', 'airline', 'name', 'text', 'tweet_coord', 'tweet_created', 'tweet_location', 'negativereason']]

# Print number of tweets
print()
print("Number of Tweets is: " + str(len(tweets)))

########## Compute VADER scores ###################

i=0 
compval1 = []  
while (i<len(tweets)):
    k = analyser.polarity_scores(tweets.iloc[i]['text'])
    compval1.append(k['compound'])
    i = i+1
    

compval1 = np.array(compval1)

# Add new column called VADER score with computed values
tweets['VADER Score'] = compval1
#tweets.to_csv("newTweets.csv", index=False)
outfile = open('newTweets.csv', 'w')
tweets.to_csv(outfile, index=False)

####### Using VADER Scores for VADER Score column #################
i = 0

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
outfile = open('finalOutput.csv', 'w')
tweets.to_csv(outfile, index=False)
outfile.close()

################### Prediction Accuracy #####################

correctPredictions = tweets[tweets['airline_sentiment'] == tweets['vader_airline_sentiment']]
print("Percentage of correct predictions: " + str(len(correctPredictions)/len(tweets)))
#exit()
