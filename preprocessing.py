import re
import os
import math
import glob
import string
import pickle
import numpy as np
import pandas as pd
import urllib.request
from tqdm import tqdm
from nltk import bigrams
from functools import partial
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup,SoupStrainer
from multiprocessing import Pool , cpu_count
from sklearn.neighbors import KNeighborsClassifier
from nltk.collocations import BigramCollocationFinder

#### Module 1
### Connect to the CSI courses website
url = 'https://catalogue.uottawa.ca/en/courses/csi/'
response = urllib.request.urlopen(url)
soup_packetpage = BeautifulSoup(response, 'lxml')
items = soup_packetpage.findAll("div", class_ = "courseblock")

lemmat = WordNetLemmatizer()
stemmer = PorterStemmer()

#### Module 3
def preprocessing(str_info):
    tokenized_word = word_tokenize(str_info)

    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()

    sp_removed_word = []
    # Stopword Removal
    stop_words = set(stopwords.words('english')\
                     +stopwords.words('french')\
                     +list(string.ascii_lowercase)\
                     +[ str(i) for i in range(0,10)])
    for word in tokenized_word:
        # Lemmatization
        word = wordnet_lemmatizer.lemmatize(word)

        # Stemming - Porter Stemmer
        word = porter_stemmer.stem(word)

        if word.lower() not in stop_words:
            sp_removed_word.append(
                word.lower())

    return sp_removed_word

# ========================
# Clean the text data that in the training set
# ========================
# ref: https://medium.com/@chaimgluck1/have-messy-text-data-clean-it-with-simple-lambda-functions-645918fcc2fc

def cleaning(df):

#     # drop the ID column
#     df = df.drop(['docID'],axis=1)

    # combine name and description into one column
    df['text'] = df['description']+df['name']

    # apply preprocessing(aString) function in text
    df.text = df.text.apply(lambda x: preprocessing(x))
    return df


def corpus_preprocessing(items):
    '''input a list of result.Tags and return a list of dictionarized courses'''
    # 1. Save the document ID, name(e.g., Database I)
    # and the corresponding description into a dict - course:

    # 2. Save all courses info into a list of dict - courses
    courses = []
    docID = 0
    for tag in items:
        # Eliminate French Courses and the unit info
        regex = r'(?P<code>\w{3}\xa0\d{1}[1|2|3|4|9|0]\d{2})\s(?P<name>.*)'

        course = re.match(regex ,tag.find('strong').text.lower())
        if course:
            course = course.groupdict()
            course['docID'] = docID
            course['name'] = re.sub(r'\([^)]*\)', '', course['name'])
            course['description'] = tag.find('p',{'class':'courseblockdesc noindent'})
            if course['description']:
                course['description'] = course['description'].text
                courses.append(course)
            docID += 1
    return courses

def four_step_processing(curr):
    # Tokenization
    #   Concatenate course name and the description into one string
    unneed_key = [ 'code', 'docID']
    keys = list(curr.keys())
    for key in unneed_key:
        if key in keys:
            keys.remove(key)

    str_info = ""
    for key in keys:
        str_info += curr[key]

    #   Remove the punctuation from the string
    for c in string.punctuation:
        str_info = str_info.replace(c,"")
    return preprocessing(str_info)


def all_occurred_word(courses, reuters):
    if os.path.exists('words_dicts.dicts'):
        with open('words_dicts.dicts', 'rb') as f:
            result = pickle.load(f)
        return result

    list_of_word_lists = []
    for i in tqdm(courses):
        list_of_word_lists.append(four_step_processing(i))

    for reuter in tqdm(reuters):
        list_of_word_lists.append(
            four_step_processing(reuter))

    # Combine all sublists into one list
    one_list_word = []
    for word_list in tqdm(list_of_word_lists):
        one_list_word += word_list
    result = [ list_of_word_lists , one_list_word]
    with open('words_dicts.dicts', 'wb') as f:
        pickle.dump(result, f)
    return result

def parse_reuter(reuters):

    topicList = []
    titleList = []
    bodyList = []

    for i in reuters:
        # only select the one with topic(s)
        if(i.topics.get_text()!=''):

            # join multiple topics with space
            oneStr = ' '

            # create a list to save the topics
            res = []
            for j in i.topics:
                res.append(j.get_text().lower())

            currTopic = oneStr.join(res)

            topicList.append(currTopic)
        else:
            topicList.append('Nan')

        if(i.title):
            currTitle = i.title.get_text()

            # drop punctuation and numbers
            # make everything lowercase
            currTitle = currTitle.translate(
                str.maketrans('','','1234567890'))
            currTitle = currTitle.translate(
                str.maketrans('','',string.punctuation))

            titleList.append(currTitle.lower())
        else:
            titleList.append('Nan')

        if(i.body):
            currBody = i.body.get_text()

            # subitle non alpahanumeric character with ' '
            currBody = re.sub('[^0-9a-zA-Z]+', ' ', currBody)

            # drop punctuation and numbers
            # make everything lowercase
            currBody = currBody.translate(
                str.maketrans('','',string.punctuation))
            currBody = currBody.translate(
                str.maketrans('','','1234567890'))

            # lower case the processing
            bodyList.append(currBody.lower())
        else:
            bodyList.append('Nan')

    zipped_lists = zip(topicList, bodyList, titleList)

    dic_list = [{'topic': topic,
                 'description': body,
                 'name': title}
                for topic, body, title in zipped_lists]
    return dic_list

def manyTopics(df):
    # Deal with the document with more than one topics
    #
    # Original:
    # row1:  D1   t1, t2, t3
    #
    # New:
    # row1:  D1   t1
    # row2:  D1   t2
    # row3:  D1   t3
    # Get the number of rows
    rs = df.shape[0]

    for i in range(rs):

        topic = df.iloc[i]['topic']

        # Find the row with multiple topics
        if(' 'in topic):

            # Keep other data the same
            des = df.iloc[i]['description']
            dID = df.iloc[i]['docID']
            name = df.iloc[i]['name']

            # Split the long string into a list
            topicList = topic.split()

            for j in topicList:

                newOne = {'description':des,
                         'docID':dID,
                         'name':name,
                          'topic':j
                         }

                # Append the new row to the end
                df = df.append(newOne,ignore_index=True)

                # Drop the old row
                df = df.drop(i)

    df = df.sort_values(by=['docID'])

    return df

def document_reuter_parse(docs):
    list_of_reuters = []
    for doc in docs:
        curr = open(doc.encode('UTF-8'),'r',errors='ignore')
        curr_data = curr.read()
        soup = BeautifulSoup(curr_data,'html.parser')

        # find the high level block
        doc = soup.find_all('reuters')

        list_of_reuters += parse_reuter(doc)

    return list_of_reuters

def topTerms(train_df):
    train_rows = train_df.shape[0]
    allTrainTerms = []

    # concatenate all text rows into one list
    for i in range(train_rows):
        currTermList = train_df.iloc[i]['text']
        if currTermList != -1 :
            allTrainTerms += currTermList


    # sizeAllTerms: 967,103
    sizeAllTerms = len(allTrainTerms)


    # See the number of unique terms in this list: 68,616
    numOfUniques = len(set(allTrainTerms))

    # Find the most common word
    count = Counter(allTrainTerms)

    # return a list of tuples: (term, the number of occurrence in the whole df )
    termFreqs = count.most_common(sizeAllTerms)

    # the number of terms occurrs more than 100 times: 1,234
    morethan100 = 0

    for i in termFreqs:
        if i[1] > 100 :
            morethan100 += 1

    topTerms = count.most_common(morethan100)

    return topTerms

def count_words(word,index):
    if index == "-1":
        return 0
    return index.count(word)

def insertVS(source, terms_dict):
    listOfTerms = list(map(lambda x: x[0] ,terms_dict ))
    result = dict(zip(listOfTerms,[[] for i in listOfTerms]))

    df = pd.DataFrame(result)
    oc = []

    for word in listOfTerms:
        oc.append(source.text.iloc[0].count(word))

    cnt = np.vectorize(count_words,excluded=['index'])

    occurance = source.text.apply(
        lambda x: np.array(
            cnt(listOfTerms,index=x))).values

    occurance = np.vstack(occurance)

    for (i,col) in enumerate(df.columns):
        df[col] = occurance[:,i]

    df['topic'] =  source.topic.values
    df['docID'] = source.docID.values

    return df

def reuters_parser():
    final_result = []

    documents = glob.glob('reuters/*.sgm')
    partion_size = len(documents)
    documents = [
        documents[k:k + partion_size//4]
        for k in range(0,partion_size , partion_size//4)]

    with Pool(processes=len(documents)) as pool:
        result = pool.map(document_reuter_parse,documents)

    for i in result:
        final_result += i

    for (docID,item) in enumerate(final_result):
        item['docID'] = docID

    return final_result

def topic_classification_datasets(retures):
    train_dic, predict_dic = [] , []
    for i in retures:
        if i['topic'] == 'Nan':
            predict_dic.append(i)
        else:
            train_dic.append(i)

    if os.path.exists('topic_clas_train.csv') and os.path.exists('topic_clas_predict.csv'):
        train_df = pd.read_csv('topic_clas_train.csv',engine='python')
        predict_df = pd.read_csv('topic_clas_predict.csv',engine='python')
        return predict_df , train_df

    train_df = pd.DataFrame(train_dic)
    predict_df = pd.DataFrame(predict_dic)

    # topicization the data frame
    train_df = manyTopics(cleaning(train_df))

    # handling null cases
    train_df.fillna(-1,inplace=True)
    train_df.to_csv('topic_clas_train.csv',index=False) , predict_df.to_csv('topic_clas_predict.csv',index=False)
    return predict_df , train_df


def load_train_df():
    if os.path.exists('topic_clas_train.csv') and os.path.exists('topic_clas_predict.csv'):
        print("preivously preprocessed csv found loading into memory ")

        train_df = pd.read_csv('topic_clas_train.csv',engine='python')
        train_df.replace([r'\[',r'\]',r"'",' '], ['','','',''],inplace=True,regex=True)
        train_df.text = train_df.text.apply(
            lambda text: text.split(',') if text != '-1' else text)

        predict_df = pd.read_csv('topic_clas_predict.csv',engine='python')
        predict_df.replace([r'\[',r'\]',r"'",' '], ['','','',''],inplace=True,regex=True)
        predict_df.text = predict_df.text.apply(
            lambda text: text.split(',') if text != '-1' else text)

        return train_df , predict_df
    raise FileNotFoundError("top_clas_trian.csv and topic_clas_predict.csv not found")


def read_preprocessed_docs():
    if os.path.exists('documents.dicts'):
        print("previous documents found loading into memory")
        with open('documents.dicts', 'rb') as f:
            result = pickle.load(f)
        return result[0] , result[1]
    else:
        print("Preprocessing the reuter and course data")
        courses = corpus_preprocessing(items)
        reuters = reuters_parser()
        # saving the current file read processes
        with open('documents.dicts', 'wb') as f:
            pickle.dump([courses, reuters], f)
        return courses , reuters

def knn_classifier(train,reuters):
    top_terms = topTerms(train)

    train = insertVS(train, top_terms)
    test  = insertVS(predict, top_terms)

    train_x = train[train.columns[:-2]]
    train_y = train.topic

    k = round(math.sqrt(train.shape[0]))

    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier = classifier.fit(train_x , train_y)
    predict.topic = classifier.predict(
        test[test.columns[:-2]].values)

    for (docid,topic) in zip(predict.docID,predict.topic):
        reuters[docid]['topic'] = topic

@np.vectorize
def convert_greater(val):
    return 1 if int(val) > 0 else 0

def count_zeros(df, i , j):
    cnt = np.count_nonzero(
        np.apply_along_axis(
            lambda x: True if x[j] == 1 and x[i] ==1 else False,
            -1 , df))
    return cnt

def compute_jaccard(thesaurus, data_val, dnomintor, item):
    print("Preprocessing from [{}:{}]".format(item[0],item[-1]))
    for i in item:
        for j in range(thesaurus.shape[0]):
            if(i != j):
                if max(dnomintor[i],dnomintor[j]) > 0:
                    thesaurus[i,j] = count_zeros(data_val,i,j) / \
                        max(dnomintor[i],dnomintor[j])
                else:
                    thesaurus[i][j] = 0
            else:
                continue
    print("Preprocessing from [{}:{}] completed".format(item[0],item[-1]))

# Module 9a Automatic Thesaurus Construction
def build_thesaurus():
    if os.path.exists('pre_th.csv'):
        datasets = pd.read_csv('pre_th.csv',engine='python')
    else:
        train , test = load_train_df()
        datasets = pd.concat([train,test],ignore_index=True)
        datasets = insertVS(datasets, topTerms(datasets))

    thesaurus = np.identity(len(datasets.columns[:-2]))
    data_val = convert_greater(
        datasets[
            datasets.columns[:-2]].values)

    dicts = np.identity(len(datasets.columns[:-2]))
    dicts = dict(zip( datasets.columns[:-2] , dicts))

    thesaurus = pd.DataFrame(dicts)
    thesaurus.set_index(datasets.columns[:-2],inplace=True)

    dnomintor = np.sum(data_val,-1)

    for i , row in enumerate(tqdm(datasets.columns[:-2])):
        for j , col in enumerate(datasets.columns[i+1:-2]):
            if max(int(dnomintor[i]) , int(dnomintor[j])) > 0 and i != j:
                thesaurus[col][row] = len(
                    set(np.argwhere( data_val[i,:] > 0).T.tolist()[0])  & \
                    set(np.argwhere( data_val[j,:] > 0).T.tolist()[0])
                    )\
                    / max(int(dnomintor[i]) , int(dnomintor[j]))
    thesaurus.to_csv('./thesaurus.csv')

    # brute force way

    # zeos = np.zeros(thesaurus.shape[0])
    # iterator = [ i for i in range(thesaurus.shape[0])]
    # item = [
    #     iterator[k : k + thesaurus.shape[0]//4]
    #     for k in range(0,thesaurus.shape[0],thesaurus.shape[0]//4)]


    # func = partial(compute_jaccard, thesaurus, data_val, dnomintor)
    # with Pool(processes=len(item)) as pool:
    #     pool.map(func,item)

    return data_val , thesaurus



courses , reuters = read_preprocessed_docs()
thrasure = pd.read_csv ('./thesaurus.csv',index_col=[0])
all_topics = pd.read_csv('topic_clas_train.csv', engine='python').topic
list_of_word_lists ,one_list_word = all_occurred_word(courses, reuters)
