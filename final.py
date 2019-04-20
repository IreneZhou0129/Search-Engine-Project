import re 
import glob
import string 
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup,SoupStrainer
from nltk import bigrams
from nltk.collocations import BigramCollocationFinder
from preprocessing import courses , all_topics
from preprocessing import preprocessing as preprocessing
# Collect a list of all Reuters documents
documents = glob.glob('reuters/*.sgm')

# Create a list of dictionaries for the current reuter file
# e.g., [{'text':xxx, 'title':xxx}, ...]
def build_dict(document):

    # Read the file
    curr = open(document.encode('UTF-8'),'r',errors='ignore')
    curr_data = curr.read()
    soup = BeautifulSoup(curr_data,'html.parser')

    topic_list = [i.string for i in soup.findAll("topics")]
    text_list = [i.string for i in soup.findAll("body")]
    title_list = [i.string for i in soup.findAll("title")]

    # zipped_lists = zip(topic_list,text_list,title_list)
    # dic_list = [{'topic': topic, 'text': text, 'title': title} 
    #     for topic,text,title in zipped_lists]
        
    zipped_lists = zip(text_list,title_list)

    dic_list = [{'description': text, 'name': title} 
        for text,title in zipped_lists]
    
    # Replace all None values with 'NaN'
    for i in dic_list:
        for k,v in i.items():
                if v is None:
                    i[k] = "NaN"

    return dic_list

# input: the list of all reuter documents: [r1, r2, r3, ...]
# output: a list of all reuter dictionaries, each has a new key: docID
def rt_corpus_preprocessing(docs):

    list_of_lists = []
    docID = 0
    dictionary = {}

    for i in range(len(docs)): 
        
        # read the file
        document = docs[i]
        # curr_dic_list: [{'topic':xxx, 'text':xxx, 'title':xxx}, ...]
        curr_dic_list = build_dict(document)
        list_of_lists.append(curr_dic_list)
    
    # Combine all sublists into one list, size is 20841
    flatten = [item for sublist in list_of_lists for item in sublist]
    
    # Add new key: docID 
    for i in flatten:
        i["docID"] = docID
        docID += 1
        
    return flatten

def single_text_prep(text):

    # Substitute multiple spaces with single white space 
    processed_txt = ' '.join(text.split())

    # Remove the punctuation from the string
    for c in string.punctuation:
        processed_txt = processed_txt.replace(c,"")

    # Remove digits
    processed_txt = re.sub(r"\d+", "", processed_txt)

    # Preprocessing
    processed_txt = preprocessing(processed_txt)

    return processed_txt


# ===============================================================
# Bigram Language Model
# ===============================================================
# ref: https://bit.ly/2TRJZko 
def bigram_model(list_of_dicts):
    
    # Concatenate all descriptions into one list 
    one_list_word = []
    for i in list_of_dicts:
        curr_txt = i["description"] 
        curr_word_list = single_text_prep(curr_txt)
        one_list_word.append(curr_word_list)

    
    # Flat the list of lists
    one_list_word = [item for sublist in one_list_word for item in sublist]

    finder = BigramCollocationFinder.from_words(one_list_word)
    finder.apply_freq_filter(5)
    bigram_model = finder.ngram_fd.items()
    bigram_model = sorted(finder.ngram_fd.items(), key=lambda item: item[1],reverse=True)  
    return bigram_model
    # np.save("bigram_model.npy",bigram_model)

# # load npy file
# t1=time.time()
# array_reloaded = np.load('bigram_model.npy')
# t2=time.time()

# print(array_reloaded.shape)
# # (46878, 2)


if __name__ == "__main__":
    # flatten = rt_corpus_preprocessing(documents)
    model = bigram_model(courses)
    result = {}
    for ((k1 , k2), count) in tqdm(model): 
        if k1 not in result.keys():
            result[k1] = {}
        result[k1][k2] = count


