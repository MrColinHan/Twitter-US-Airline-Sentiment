import csv
import nltk
from nltk.tokenize import word_tokenize



def read_csv(filedir, listname):
    file = open(filedir)
    reader = csv.reader(file)
    for row in reader:
        listname.append(row)


