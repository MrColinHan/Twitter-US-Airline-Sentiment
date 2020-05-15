# Web Mining Project
## Topic: Twitter-US-Airline-Sentiment

### 1. Naive Bayes Classifier: 
 #### Procedure:
    (naive_bayes_classifier.py)
    a. tokenize tweet text
    b. normalize text
    c. remove noise
    d. calculate word density
    e. prepare data for training the classifier
    f. build&test model
    g. calculate accuracy
    h. collect new data: extract airline-related tweets from raw tweets
        (parse_new_tweets.py)
    i. deploy the trained model on new data
    
    Files: 
        negative_tweets.csv: for model training
        positive_tweets.csv: for model training
        parsed_tweet_0.csv: new data for predicting
        parsed_tweet_0_(with_prediction).xlsx: new data with predicted sentiment values(predicted with a 92.1% accuracy model)

### 2. GeoSpatial Analysis: 
 #### Input file: 
    input_Tweets.csv
 #### Count location frequency and its output:
    tweet_location_count.py 
    location counts.csv
 #### Edit the output and analyze it in Power BI
    edited location counts.csv
    Power BI.pbix
 #### Export the visualization as images
    US map.png, US_Alaska.png, US_Hawaii.png, world_map.png

### 3. Sentiment Word Cloud Analysis: 
 #### Input files:
    negative texts.csv, positive texts.csv
 #### Tool: 
    wordart.com
 #### Output Images and word weights:
    negative_wordCloud.png, positive_wordCloud.png
    (weight)negative texts.csv, (weight)positive texts.csv
 #### Use NLTK to add tags and build new wordcloud for adj&verb:
    (weight&tag)negative texts.csv, (weight&tag)positive texts.csv
    (verb&adj)negative_wordCloud.png, (verb&adj)positive_wordCloud.png

