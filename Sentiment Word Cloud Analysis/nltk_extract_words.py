import csv
import nltk
from nltk.tokenize import word_tokenize


def read_csv(filedir, listname):
    file = open(filedir)
    reader = csv.reader(file)
    for row in reader:
        listname.append(row)


def write_csv(x, y):  # write list x into file y
    with open(y,'w+') as file:
        wr = csv.writer(file, dialect='excel')
        wr.writerows(x)
    file.close()


def main():
    input_list = []
    read_csv("(weight)negative texts.csv", input_list)
    '''
    tokens_list = []
    for row in input_list[1:]:  # exclude first row
        tokens_list.append(row[0])

    pos_tag_list = nltk.pos_tag(tokens_list)
    print(pos_tag_list[:8])
    
    jj_list = []
    for tuple in pos_tag_list:
        if tuple[1] in ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            jj_list.append(tuple)
    print(jj_list)

    print(nltk.pos_tag(['Gets']))
    '''
    for i in range(len(input_list)):
        input_list[i].append((nltk.pos_tag([input_list[i][0]]))[0][1])

    write_csv(input_list, "(weight&tag)negative texts.csv")
    #print(nltk.pos_tag(['Best']))

main()


