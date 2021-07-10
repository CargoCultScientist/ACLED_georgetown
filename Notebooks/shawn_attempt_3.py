import pandas as pd
from pandas import DataFrame
import re
import csv
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
name_data = "/Users/admin/desktop/2019-05-01-2021-05-29-Ethiopia copy.csv"
df = pd.read_csv(name_data, sep= ',', header=0)
df1 = df[(df.event_type == 'Protests')]
df2 = df1.drop(df1.loc[:, 'data_id':'source_scale'].columns, axis = 1)
df3 = df2.drop(df2.loc[:, 'fatalities':'iso3'].columns, axis = 1)
protester_text = list(df3['notes'])
stopwords = nltk.corpus.stopwords.words('english')

'''
things to do:

- need to take out all location data type words out, location seems to add no value to what we are trying to do...??
- need to build out rest of code to take on the remaining type events...
'''

'''
remove all words less than or equal to 4 charaters
'''
def remove_4charwrds(text_in):
    text = str(text_in)
    shortword = re.compile(r'\W*\b\w{1,4}\b')
    return shortword.sub('', text)
'''
remove stopwords
'''
def remove_stopwords(data):
    output_array=[]
    new_stpwrds = ['january','february','march', 'april','may','june','july', 'august',
                   'september','october','november','december']
    stopwords.extend(new_stpwrds)
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array
'''
remove all numbers (dates too)
'''
def remove_num(list):
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list]
    return list  
 '''
 remove any type brackets
 '''
def brackets_rem(list):
    list = [re.sub('[\(\[].*?[\)\]]', '', i) for i in list]
    return list 

'''
process protester text
'''
output_protests_pre1 = remove_stopwords(protester_text)
output_protests_pre2 = remove_num(output_protests_pre1)
output_protests_pre3 = brackets_rem(output_protests_pre2)
output_protests_final = remove_4charwrds(output_protests_pre3)

# type(output_protests_final)
# print(output_protests_final)

stopwords = stopwords.words('english')
name_data = "/Users/admin/desktop/2019-05-01-2021-05-29-Ethiopia copy.csv"
df = pd.read_csv(name_data, sep= ',', header=0)
df1 = df[(df.event_type == 'Battles')]
df2 = df1.drop(df1.loc[:, 'data_id':'source_scale'].columns, axis = 1)
df3 = df2.drop(df2.loc[:, 'fatalities':'iso3'].columns, axis = 1)
battle_text = list(df3['notes'])
'''
process battles text
'''
output_battles_pre1 = remove_stopwords(battle_text)
output_battles_pre2 = remove_num(output_battles_pre1)
output_battles_pre3 = brackets_rem(output_battles_pre2)
output_battles_final = remove_4charwrds(output_battles_pre3)

# type(output_battles_final)
# print(output_battles_final)

'''
read and write in data to local machine
'''
protest_text_file = open("protestinfo.txt", "w")
n = protest_text_file.write(output_protests_final)
protest_text_file.close()
file = open('protestinfo.txt', 'r')
protest_doc = file.read()
protest_txt = str(protest_doc)

battle_text_file = open("battleinfo.txt", "w")
f = battle_text_file.write(output_battles_final)
battle_text_file.close()
file1 = open('battleinfo.txt', 'r')
battle_doc = file1.read()
battle_txt = str(battle_doc)


'''
populate corpus(s):
'''
corpus_total = [protest_txt, battle_txt]
# corpus_total = [protest_txt_lemma, battle_txt_lemma]
# corpus_protest = file
# corpus_battle = file1

'''
vectorize data:
'''
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus_total)
# vectorizer.fit(corpus_total)
# total_vocab = vectorizer.vocabulary_

vectorizer = TfidfVectorizer()
Y = vectorizer.fit_transform(corpus_total)
# vectorizer.fit(corpus_total)
# total_vocab = vectorizer.vocabulary_

'''
seperate protest vocab creation(ex):
'''
# vectorizer_protest = CountVectorizer()
# P = vectorizer_protest.fit_transform(corpus_protest)
# vectorizer_protest.fit(corpus_protest)
# protest_vocab = vectorizer_protest.vocabulary_

'''
seperate battles vocab creation(ex):
'''
# vectorizer_battles = CountVectorizer()
# B = vectorizer_battles.fit_transform(corpus_battle)
# vectorizer_battles.fit(corpus_battle)
# battle_vocab = vectorizer_battles.vocabulary_

'''
create dataFrame(s)
standard doc_term freq matrix across all docs (total):
'''
# df1 = pd.DataFrame(Y.toarray().transpose(),
#                    index=vectorizer1.vocabulary_)

df1 = pd.DataFrame(Y.toarray().transpose(),
                   index=vectorizer.get_feature_names())
'''
tfidf total df:
'''
# df2 = pd.DataFrame(X.toarray().transpose(),
#                    index=vectorizer.vocabulary_)

df2 = pd.DataFrame(X.toarray().transpose(),
                   index=vectorizer.get_feature_names())

# print(vectorizer.get_feature_names())
# print(corpus_total)
'''
set column names:
'''
df1.columns = ['protest_doc','battle_doc']
df2.columns = ['protest_doc','battle_doc']
print('Tfid_table')
print(df1)
print('doc_term_freq_matrix')
print(df2)
# print(total_vocab)
# print(protest_vocab)
# print(battle_vocab)
# print(X.shape)
# print(X.toarray())
# print(Y.shape)
# print(Y.toarray())
# print(stopwords)
'''
take a look at the tables made(csv)
'''
df1.to_csv("/Users/admin/desktop/tfidf_takealook.csv")
df2.to_csv("/Users/admin/desktop/doc_term_takealook.csv")

'''
find max values
'''
column1 = df1["protest_doc"]
max_value1 = column1.max()
print('max tfidf value - protest vocab:')
print(max_value1)
column2 = df1["battle_doc"]
max_value2 = column2.max()
print('max tfidf value - battle vocab:')
print(max_value2)


