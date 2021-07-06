"""
If you put this sequentially into a notebook you should get to data sets using 
protest and battle data sets/documents. Change the name_data to a file path on your 
machine. The two 'takealook' cvs files generated at the end need to be changed to 
your desktop if you want to see kinda the cleaned datasets being created. Not 100% 
sure where to go from here, if this is remotely right. If it is, it can easily be scaled 
to do all topics, not just protests and battles...
"""

import pandas as pd
import re
import csv
import nltk
from string import digits
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
'''
import data
'''
name_data = "/Users/admin/desktop/2019-05-01-2021-05-29-Ethiopia copy.csv"
df = pd.read_csv(name_data, sep= ',', header=0)
'''
clean data
'''
df1 = df[(df.event_type == 'Protests')]
df2 = df1.drop(df1.loc[:, 'data_id':'source_scale'].columns, axis = 1)
df3 = df2.drop(df2.loc[:, 'fatalities':'iso3'].columns, axis = 1)
protester_text = list(df3['notes'])
stopwords = set(stopwords.words('english'))
data = protester_text
def remove_stopwords(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array

output_protests_pre = remove_stopwords(data)
# print(output_protests)

def remove_num(list):
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list]
    return list  

output_protests = remove_num(output_protests_pre)


df = pd.read_csv(name_data, sep= ',', header=0)
df1 = df[(df.event_type == 'Battles')]
df2 = df1.drop(df1.loc[:, 'data_id':'source_scale'].columns, axis = 1)
df3 = df2.drop(df2.loc[:, 'fatalities':'iso3'].columns, axis = 1)
battle_text = list(df3['notes'])
stopwords = set(stopwords.words('english'))
data = battle_text
output_battles_pre = remove_stopwords(data)
output_battles = remove_num(output_battles_pre)

with open('protestinfo.txt', 'w') as file:
    for item in output_protests:
        file.write("%s\n" % item)
with open('battleinfo.txt', 'w') as file1:
    for item in output_battles:
        file1.write("%s\n" % item)
        
file = open('protestinfo.txt', 'r')
protest_doc = file.read()
protest_txt = str(protest_doc)
        
file1 = open('battleinfo.txt', 'r')
battle_doc = file1.read()
battle_txt = str(battle_doc)

'''
populate corpus(s):
'''

corpus_total = [protest_txt, battle_txt]
# corpus_protest = file
# corpus_battle = file1

'''
vectorize data:
'''
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus_total)
vectorizer.fit(corpus_total)
total_vocab = vectorizer.vocabulary_


vectorizer1 = TfidfVectorizer()
Y = vectorizer1.fit_transform(corpus_total)


'''
seperate protest vocab creation:
'''
# vectorizer_protest = CountVectorizer()
# P = vectorizer_protest.fit_transform(corpus_protest)
# vectorizer_protest.fit(corpus_protest)
# protest_vocab = vectorizer_protest.vocabulary_

'''
seperate battles vocab creation:
'''
# vectorizer_battles = CountVectorizer()
# B = vectorizer_battles.fit_transform(corpus_battle)
# vectorizer_battles.fit(corpus_battle)
# battle_vocab = vectorizer_battles.vocabulary_

'''
create dataFrame(s)
standard doc_term freq matrix across all docs (total):
'''
df1 = pd.DataFrame(Y.toarray().transpose(),
                   index=vectorizer1.vocabulary_)
'''
tfidf total df:
'''
df2 = pd.DataFrame(X.toarray().transpose(),
                   index=vectorizer.vocabulary_)

'''
set column names:
'''
df1.columns = ['protest_doc','battle_doc']
df2.columns = ['protest_doc','battle_doc']
print('Tfid_table')
print(df1)
print('doc_term_matrix')
print(df2)
# print(total_vocab)
# print(protest_vocab)
# print(battle_vocab)
# print(X.shape)
# print(X.toarray())
# print(Y.shape)
# print(Y.toarray())
# print(stopwords.words('english'))
df1.to_csv("/Users/admin/desktop/tfidf_takealook.csv")
df2.to_csv("/Users/admin/desktop/doc_term_takealook.csv")


column1 = df1["protest_doc"]
max_value1 = column1.max()
print('max tfidf value - protest vocab:')
print(max_value1)
column2 = df1["battle_doc"]
max_value2 = column2.max()
print('max tfidf value - battle vocab:')
print(max_value2)
