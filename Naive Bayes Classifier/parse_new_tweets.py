"""
@author: Changze Han
"""

import re
from re import compile
import pandas as pd
import csv
'''
    This program extracts airline-related tweets out of a raw tweet dataset. 
    
    Raw tweet data frame columns: 
        | Null | tweet_id | name | tweet_created | text | tweet_location | tweet_coord | ...(more Unnamed)...
        
    Desired output data frame columns: 
        | tweet_id | airline | name| text | tweet_coord | tweet_created | tweet_location | 
        
'''
# =============================================================================
input_file_dir = r"/Users/Han/Downloads/web project data/tweet_1.csv"
output_file_dir = r"/Users/Han/Downloads/web project data/parsed_tweet_1.csv"

chunk_size = 500000  # for each data frame
# =============================================================================
STRING_MATCHER = ["@delta ", "#delta ",
                  "@unitedairlines", "#unitedairlines", "@united ", "#united ",
                  "@virginamerica", "#virginamerica",
                  "@southwestair", "#southwestair",
                  "@jetblue", "#jetblue",
                  "@usAirways", "#usAirways",
                  "@americanair", "#americanair"]  # append more airlines here
airline_name_dict = {"@delta ": "Delta", "#delta ": "Delta",
                     "@unitedairlines": "United", "#unitedairlines": "United", "@united ": "United", "#united ": "United",
                     "@virginamerica": "Virgin America", "#virginamerica": "Virgin America",
                     "@southwestair": "Southwest", "#southwestair": "Southwest",
                     "@jetblue": "JetBlue", "#jetblue": "JetBlue",
                     "@usAirways": "US Airways", "#usAirways": "US Airways",
                     "@americanair": "American", "#americanair": "American"}  # American, Delta, Southwest, United, US Airways, Virgin America,
# prepare output data frame
output_column_names = ["tweet_id", "airline", "name", "text", "tweet_coord", "tweet_created", "tweet_location"]
output_df = pd.DataFrame(columns=output_column_names)


def read_chunk(d, n):  # d is firectory, n is number of lines for each chunk
    for chunk in pd.read_csv(d, chunksize=n):
        print(chunk)


def write_csv(x, y):
    with open(y,'w+') as file:
        wr = csv.writer(file, dialect='excel')
        wr.writerows(x)
    file.close()


def check_string(string):
    # extract string from the given string, returns the first match if there are multiple matches
    return STRING_MATCHER.findall(string)[0]


def main():
    global output_df
    chunk_count = 1
    for chunk in pd.read_csv(input_file_dir, chunksize=chunk_size):
        #chunk_copy = chunk[chunk.columns[0:7]]  # only keep the first 7 columns
        #print(chunk)
        for i in chunk.index:  # iterate each row
            #print(str(chunk["text"][i]))
            for keyword in STRING_MATCHER:
                if keyword in str(chunk["text"][i]).lower():
                #if re.search(keyword, chunk["text"][i], re.IGNORECASE):  # check key word
                    print(str(chunk["text"][i]))
                    output_df = output_df.append({"tweet_id": chunk["tweet_id"][i],
                                                 "airline": airline_name_dict[keyword],
                                                  "name": chunk["name"][i],
                                                  "text": chunk["text"][i],
                                                  "tweet_coord": chunk["tweet_coord"][i],
                                                  "tweet_created": chunk["tweet_created"][i],
                                                  "tweet_location": chunk["tweet_location"][i]},
                                                 ignore_index=True, sort=False)
                    break
        print(f"Done: chunk {chunk_count}: {len(chunk)} tweets")
        chunk_count += 1
    # write output csv file
    output_df.to_csv(output_file_dir, index=False)
    print(f"Output Written: extracted {len(output_df)} tweets")


main()

