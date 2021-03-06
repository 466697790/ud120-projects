#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
print(type(enron_data), len(enron_data))
f_count = len(enron_data['METTS MARK'])
print(f_count)
enron_data_list = list(enron_data)
print(enron_data_list)
print(len(enron_data_list))

# count = 0
# for p in enron_data_list:
#     if enron_data[p]['poi'] == 1:
#         count += 1

# print(count)

# stock = enron_data['PRENTICE JAMES']['total_stock_value']
# print(stock)
# email_count = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
# print(email_count)
# value = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
# print(value)

print(enron_data['METTS MARK'])
count = 0
for p in enron_data_list:
    # if enron_data[p]['salary'] != 'NaN':
    if enron_data[p]['email_address'] != 'NaN':
        count += 1

print(count)











