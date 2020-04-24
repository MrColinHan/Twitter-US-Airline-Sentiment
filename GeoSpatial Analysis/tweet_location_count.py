import csv


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
    read_csv("Kaggle_Tweets.csv", input_list)

    address_column_header = 'tweet_location'
    header_index = input_list[0].index(address_column_header)  # input_list[0] is the 1st row
    print(header_index)

    address_count_dict = {}
    for row in input_list[1:]:  # exclude first row
        if row[header_index] != '':
            if row[header_index] not in address_count_dict:
                address_count_dict[row[header_index]] = 1
            else:
                address_count_dict[row[header_index]] += 1
    print(address_count_dict)

    output_list = [list(address_count_dict.keys()), list(address_count_dict.values())]
    write_csv(output_list, "GeoSpatial Analysis/location counts.csv")

    '''
    Next step: 
        open the output file 'location counts.csv' in Excel
        transpose paste the first two rows into two columns
        add two headers "location" "count"
        then open the edited file in Power BI
    '''


main()



