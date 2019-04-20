import json
import re 
import string
import numpy as np
from functools import reduce
from collections import defaultdict
from pythonds.basic.stack import Stack
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocessing as sentence_preprocessing 
from preprocessing import one_list_word , list_of_word_lists, thrasure , courses , all_topics , four_step_processing
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
# Optional Module
# Spelling correction: edit distance
# ref: https://bit.ly/2T37dTt
def compute_dist(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def levenshteinDistance(s1):
    # find the most similar word to one_list_word
    min_dist = 1000
    closest = None
    if s1 not in one_list_word:
        for s2 in one_list_word:
            dist = compute_dist(s1, s2)
            if dist < min_dist:
                min_dist = dist
                closest = s2
    else:
        closest = None
        min_dist = -1

    return min_dist , closest


def dictionary_building(courses):
    # Indexing the word
    #   Create the transform
    vectorizer = TfidfVectorizer()

    #   Tokenize and build vocab
    bow = vectorizer.fit_transform(one_list_word)
    #   Summarize
    indexed_word_dict = vectorizer.vocabulary_
    # print(vectorizer.get_feature_names()),    print(indexed_word_dict),   print(vectorizer.idf_)
    return indexed_word_dict
# test case:
# indexed_word_dict = dictionary_building(courses)

#### Module 4
# arguments: courses - document collection; indexed_word_dict - dictionary
def inverted_index_construction(courses,indexed_word_dict):
    docs = [four_step_processing(i) for i in courses]
    inv_indx = defaultdict(list)
    for idx, text in enumerate(docs):
        for word in text:
            inv_indx[word].append(idx)

    result = {}
    for key, val in inv_indx.items():
        d = defaultdict(int)
        for i in val:
            d[i] += 1
        result[key] = d.items()

    #result format: 'word':dic_items([(docID, weight), (docID, weight), ...])
    return result
# test case:
# term_doc_ids = inverted_index_construction(courses,indexed_word_dict)

#### Module 5
def corpus_access(input_docIDs):
    output_docs = []
    length = len(input_docIDs)
    for i in range(length):
        curr_docID = input_docIDs.pop()
        output_docs.append(courses[i])
    return output_docs

#### Module 6
def isOperator(word):
    if(word.upper() in ['AND', 'OR', 'AND_NOT', '(', ')']):
        return True
    else:
        return False

def query_processing(user_query):
    '''Query processing: 1. transform infix format to postfix format;
                            ref: https://bit.ly/28OMA5X
                         2. deal with the wildcard '''
    # Split the query into a list of word, including the parentheses
    pattern = re.compile(r"([.()!])")
    tmp = (pattern.sub(" \\1 ", user_query)).split(" ")
    tokenList = list(filter(None,tmp))

    # Transform infix to postfix
    opStack = Stack()
    postfixList = []
    prec = {}
    prec["("] = 1
    prec["AND"] = 2
    prec["OR"] = 2
    prec["AND_NOT"] = 2
    for t in tokenList:
        if (not(isOperator(t))):
            postfixList.append(t)
        elif t == "(":
            opStack.push(t)
        elif t == ")":
            topToken = opStack.pop()
            while topToken != "(":
                postfixList.append(topToken)
                topToken = opStack.pop()
        else:
            while (not opStack.isEmpty()) and  (prec[opStack.peek()] >= prec[t]):
                postfixList.append(opStack.pop())
            opStack.push(t)

    while (not opStack.isEmpty()):
        postfixList.append(opStack.pop())
    return " ".join(postfixList)

# test case:
# user_query = "algorithm AND (data OR structure) OR (comp AND ca)"
# user_query = "perspective AND business"

# postfixQuery = query_processing(user_query)
# postfixQuery1 = query_processing(user_query1)
# print(postfixQuery1)

# print(postfixQuery):
# algorithm data structure OR AND comp prog AND OR
# ["algorithm", "data", "structure", "OR", "AND  "comp", "prog", "AND", "OR"]

def get_term_docs_list(term_doc_ids,term):
    term_docs_result = []

    # getting all of the doc_id list of tuple
    if term not in term_doc_ids.keys():
        term = 0
    else:
        # if the given term is incorrect
        pass
    term_docs = term_doc_ids.get(term)
    for k , v in term_docs:
        term_docs_result.append(k)
    return term_docs_result


# Given dict_item-type args, return two lists
def two_terms_docs_lists(term_doc_ids, term1, term2):
    term1_docs_list = []
    term2_docs_list = []
    if(type(term1)!=list and type(term2)!=list):
        term1_docs_list = get_term_docs_list(term_doc_ids, term1)
        term2_docs_list = get_term_docs_list(term_doc_ids, term2)

    elif(type(term1)!=list):
        term1_docs_list = get_term_docs_list(term_doc_ids, term1)

    elif(type(term2)!=list):
        term2_docs_list = get_term_docs_list(term_doc_ids, term2)

    else: # both of them are lists
        term1_docs_list = term1
        term2_docs_list = term2

    return term1_docs_list, term2_docs_list


def and_op(term1_docs_list, term2_docs_list):
    res = list(set(term1_docs_list) & set(term2_docs_list))
    return res

def or_op(term1_docs_list, term2_docs_list):
    res = list(set(term1_docs_list) | set(term2_docs_list))
    return res

def not_op(term1_docs_list, term2_docs_list, last_dict):
    last_doc_id = last_dict.get('docID')
    all_docs = range(0, last_doc_id)
    not_t2 = [x for x in all_docs if x not in term2_docs_list]
    res = and_op(term1_docs_list, not_t2)
    return res

def find_list_of_terms(term_doc_ids):
    list_of_terms = []
    for k, v in term_doc_ids.items():
        list_of_terms.append(k)

    return list_of_terms

def boolean_retrieval(query, courses):
    postfixQuery = query_processing(query)

    indexed_word_dict = dictionary_building(courses)
    term_doc_ids = inverted_index_construction(courses,indexed_word_dict)

    list_of_terms = find_list_of_terms(term_doc_ids)

    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    edit_dist = None
    # Turn the query into a list of word
    word_list = postfixQuery.split(" ")
    tmp = []
    for term in word_list:
        if (not isOperator(term)):
            # lemmatize
            lemmatized_term = wordnet_lemmatizer.lemmatize(term)
            # stemming
            stemmed_term = porter_stemmer.stem(term)

            dist , fix_stemmed_term = levenshteinDistance(term)

            if fix_stemmed_term:
                stemmed_term = fix_stemmed_term
                edit_dist = (dist, fix_stemmed_term)

            stemmed_word_doc_ids = get_term_docs_list(term_doc_ids, stemmed_term)

            tmp.append(stemmed_word_doc_ids)

        elif(isOperator(term)):
            t1 = tmp.pop()
            t2 = tmp.pop()
            l1, l2 = two_terms_docs_lists(term_doc_ids,t1,t2)
            if (term == "AND"):
                curr = and_op(l1, l2)
            elif(term == "OR"):
                curr = or_op(l1, l2)
            elif(term == "AND_NOT"):
                last_dict = courses[-1]
                curr = and_not_op(t1, t2, last_dict)
            tmp.append(curr)

    # ======================
    #  compute edit_dist
    # ======================
    if edit_dist:
        origin_query = ""
        for token in word_list:
            dist = compute_dist(token,edit_dist[1])
            if dist <= edit_dist[0]:
                origin_query += edit_dist[1] + " "
            else:
                origin_query += token + " "

        edit_dist = origin_query


    return reduce((lambda x , y: or_op(x,y)),tmp) , edit_dist

#### Module 7
def vsm( ourses):
    # Find idf for all word in the collection of docs
    vectorizer = TfidfVectorizer()
    bow = vectorizer.fit_transform(one_list_word)

    size_of_words = len(vectorizer.vocabulary_.keys())
    matrix = np.zeros((len(courses), size_of_words))

    for document_index, words in enumerate(list_of_word_lists):
        if words != None:
            for word in words:
                matrix[document_index,
                        vectorizer.vocabulary_[ word.lower() ] ] += 1

    word_idf = vectorizer.idf_
    tf_idf = matrix * word_idf
    return vectorizer , matrix, tf_idf

def expand_query(tokenized_query):
    cols = thrasure.columns.values
    vec_query= np.zeros(thrasure.shape[0])
    expanded_terms = []
    # founded a term matches the term
    for query_token in tokenized_query:
        matching_row = np.argwhere(cols == query_token)
        if matching_row.shape[0] != 0:
            row_index = matching_row[0][0]
            col_index = np.argpartition(thrasure.values[row_index],-2)[-2]
            col_val = cols[col_index]
            # found the term that match
            if thrasure.iloc[row_index][col_index] >= .75:
                expanded_terms.append(col_val)    
    return tokenized_query + expanded_terms
            

def vsm_query_processing(query,vectorizer, matrix ,tf_idf):

    size_of_words = len(vectorizer.vocabulary_.keys())
    query_vector = np.zeros((size_of_words))
    previous_query = sentence_preprocessing(query)
    print("before expansion ",query)
    query = expand_query(previous_query)
    print("after expansion ",query)
    # module 8 query expansion

    edit_dist = 0

    # Too slow for querying 
    for word in query:
        if word not in vectorizer.vocabulary_.keys():
            edit_dist = levenshteinDistance(word)
            query_vector[vectorizer.vocabulary_[edit_dist[1]]] += 1
        else:
            query_vector[vectorizer.vocabulary_[word]] += 1


    # # ======================
    # #  compute edit_dist
    # # ======================
    # if edit_dist:
    #     origin_query = ""
    #     for token in query:
    #         dist = compute_dist(token,edit_dist[1])
    #         if dist == edit_dist[0]:
    #             origin_query += edit_dist[1] + " "
    #         else:
    #             origin_query += token + " "

    #     edit_dist = origin_query

    query_vector =  query_vector.T * vectorizer.idf_
    rank = np.matmul(matrix , query_vector)

    if not isinstance(previous_query, list):
        previous_query = [previous_query]

    return list(reversed( np.argsort(rank))) , rank , query
# test case:
# tf = vsm("algorithm", courses)

def test():
    indexed_word_dict = dictionary_building(courses)
    term_doc_ids = inverted_index_construction(courses,indexed_word_dict)

    input_docIDs = [10, 11, 12]
    output_docs = corpus_access(input_docIDs)
    print("corpus complete")

    boolean_test_query = "algorithm AND (data OR structure) OR (comp AND ca)"
    test_postfixQuery = boolean_retrieval(boolean_test_query, courses)
    print("boolean complete")

    vsm_test_query = "algorithm data"
    test_vsm = vsm(vsm_test_query, courses)
    print("vsm complete")

def test_query_expandsion():
    # wa opec
    # manila caesar [1533, 1564],
    # tough grew [1550, 1570]
    pass

if __name__ == "__main__":
    vsm_test_query = "oil AND profit"
    # vectorizer , matrix, tf_idf = vsm(courses)
    # test_vsm, rank , dist= vsm_query_processing(vsm_test_query,vectorizer, matrix ,tf_idf)
    words = [ 'coffee','stock','oil' ,'course','corn']
    selected_terms = thrasure.columns
    for word in words:
        query = sentence_preprocessing(word)
        matching_row = np.argwhere(
            selected_terms == query[0])
        
        row_index = matching_row[0][0]
        
        col_indexes = thrasure.values[row_index].argsort()[-11:][::-1][1:]
        print("=======================================================================================================================================")
        print("[{}] being expanded to {}".format(word,selected_terms[col_indexes].tolist()))
        print("=======================================================================================================================================")