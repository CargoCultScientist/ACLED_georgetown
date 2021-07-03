'''
I can't write a function to save my life but this is my best work to date.. bascially I wanted to 
make doc-term matrix(s) where we have terms used against documents or vocab banks to showcase term 
frequency and attempt a tfidf weighted score. I only did two for protests and battle event type. I 
might have some useful code in here for whatever our final product will end up being. 
I have to redo file paths but the code fuctions and generates the two different df views.


'''
import pandas as pd
import csv
import nltk 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#------------------------------------------------------------------------------------------------------------
name_data = "/Users/admin/desktop/2019-05-01-2021-05-29-Ethiopia copy.csv"
df = pd.read_csv(name_data, sep= ',', header=0)
df1 = df[(df.event_type == 'Protests')]
df1.to_csv("/Users/admin/desktop/doc1.csv")

name_data1 = "/Users/admin/desktop/doc1.csv"
df = pd.read_csv(name_data1, sep= ',', header=0)
filter = df[['notes']]!= 0
df1 = df[filter]
df1.to_csv("/Users/admin/desktop/doc1final.csv")

'''
final transformation of protest 'notes' section into txt format:
'''
name_data = "/Users/admin/desktop/doc1final.csv"
df1 = pd.read_csv(name_data, sep= ',', header=0)
df2 = df1.drop(df1.loc[:, 'data_id':'source_scale'].columns, axis = 1)
df3 = df2.drop(df2.loc[:, 'fatalities':'iso3'].columns, axis = 1)
df3.to_csv(r'/Users/admin/desktop/protestdocfinal.txt', header=None, index=None, sep='\t', mode='a')

name_data = "/Users/admin/desktop/2019-05-01-2021-05-29-Ethiopia copy.csv"
df = pd.read_csv(name_data, sep= ',', header=0)
df1 = df[(df.event_type == 'Battles')]
df1.to_csv("/Users/admin/desktop/doc1.csv")

name_data1 = "/Users/admin/desktop/doc1.csv"
df = pd.read_csv(name_data1, sep= ',', header=0)
filter = df[['notes']]!= 0
df1 = df[filter]
df1.to_csv("/Users/admin/desktop/doc1final.csv")

'''
final transformation of battles 'notes' section into txt format:
'''
name_data = "/Users/admin/desktop/doc1final.csv"
df1 = pd.read_csv(name_data, sep= ',', header=0)
df2 = df1.drop(df1.loc[:, 'data_id':'source_scale'].columns, axis = 1)
df3 = df2.drop(df2.loc[:, 'fatalities':'iso3'].columns, axis = 1)
df3.to_csv(r'/Users/admin/desktop/battlesdocfinal.txt', header=None, index=True, sep='\t', mode='a')
#------------------------------------------------------------------------------------------------------------
file = open('/Users/admin/desktop/protestdocfinal.txt', 'r')
protest_doc = file.read()
protest_txt = str(protest_doc)

file1 = open('/Users/admin/desktop/battlesdocfinal.txt', 'r')
battle_doc = file1.read()
battle_txt = str(battle_doc)

'''
populate corpus(s):
'''
corpus_protest = [protest_txt]
corpus_battle = [battle_txt] 
corpus_total = [protest_txt, battle_txt]

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
vectorizer_protest = CountVectorizer()
P = vectorizer_protest.fit_transform(corpus_protest)
vectorizer_protest.fit(corpus_protest)
protest_vocab = vectorizer_protest.vocabulary_
# def entries_to_remove(entries, the_dict):
#     for key in entries:
#         if key in the_dict:
#             del the_dict[key]
# entries_to_remove(['on','may',
#                    '2021','21'], protest_vocab)
# for num in tuple(protest_vocab.keys()):
#     if num.isdigit():
#        del protest_vocab[num]
'''
seperate battles vocab creation:
'''
vectorizer_battles = CountVectorizer()
B = vectorizer_battles.fit_transform(corpus_battle)
vectorizer_battles.fit(corpus_battle)
battle_vocab = vectorizer_battles.vocabulary_
# entries_to_remove(['on','may',
#                    '2021','21'], battle_vocab)
# for num in tuple(battle_vocab.keys()):
#     if num.isdigit():
#        del battle_vocab[num]
'''
create dataFrame(s)
standard doc_term freq matrix across all docs (total):
'''
df1 = pd.DataFrame(Y.toarray().transpose(),
                   index=vectorizer1.get_feature_names())
'''
tfidf total df:
'''
df2 = pd.DataFrame(X.toarray().transpose(),
                   index=vectorizer.get_feature_names())

df1 = df1.drop(df1.index[range(829)])
df2 = df2.drop(df2.index[range(829)])
'''
set column names:
'''
#------------------------------------------------------------------------------------------------------------
df1.columns = ['protest_doc','battle_doc']
df2.columns = ['protest_doc','battle_doc']
df1.columns = df2.columns
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


