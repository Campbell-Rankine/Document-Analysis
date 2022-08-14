from collections import defaultdict
import pickle
import os
import numpy as np

from string_processing import *


def intersect_query(doc_list1, doc_list2):
    # TODO: you might like to use a function like this 
    # in your run_boolean_query implementation
    # for full marks this should be the O(n + m) intersection algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks
    res = []
    l1 = 0
    l2 = 0
    while l1 < len(doc_list1) and l2 < len(doc_list2):
        if doc_list1[l1] == doc_list2[l2]:
            res.append(doc_list1[l1])
            l1 += 1
            l2 += 1
        elif doc_list1[l1] < doc_list2[l2]:
            l1 += 1
        else:
            l2 += 1
    return res

def union_query(doc_list1, doc_list2):
    # TODO: you might like to use a function like this 
    # in your run_boolean_query implementation
    # for full marks this should be the O(n + m) union algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks
    res = []
    l1 = 0
    l2 = 0
    while l1 < len(doc_list1) and l2 < len(doc_list2):
        if doc_list1[l1] == doc_list2[l2]:
            res.append(doc_list1[l1])
            res.append(doc_list2[l2])
            l1 += 1
            l2 += 1
        elif doc_list1[l1] < doc_list2[l2]:
            res.append(doc_list1[l1])
            l1 += 1
        elif doc_list1[l1] > doc_list2[l2]:
            res.append(doc_list2[l2])
            l2 += 1
    if l1 < len(doc_list1):
        res = res + doc_list1[l1:]
    if l2 < len(doc_list2):
        res = res + doc_list2[l2:]
    return res

def run_boolean_query(query, index):
    """Runs a boolean query using the index.

    Args:
        query (str): boolean query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists

    Returns:
        list(int): a list of doc_ids which are relevant to the query
    """
    # TODO: implement this function
    relevant_docs = []
    Q = query.split(" ")
    if len(Q) == 1:
        return [x[0] for x in index[Q[0]]]
    for i in range(0, len(Q)-2, 2):
        if Q[i+1] == "AND": #We know our operator happens here
            l2 = [x[0] for x in index[Q[i+2]]]
            if i == 0:
                l1 = [x[0] for x in index[Q[i]]]
            else:
                l1 = relevant_docs
            assert(sorted(l1) == l1 and sorted(l2) == l2)
            add = intersect_query(l1, l2)
            relevant_docs = add
        elif Q[i+1] == "OR":
            l2 = [x[0] for x in index[Q[i+2]]]
            if i == 0:
                l1 = [x[0] for x in index[Q[i]]]
            else:
                l1 = relevant_docs
            assert(sorted(l1) == l1 and sorted(l2) == l2)
            add = union_query(l1, l2)
            relevant_docs = add
    return relevant_docs


# load the stored index
(index, doc_freq, doc_ids, num_docs) = pickle.load(open("stored_index.pik", "rb"))

print("Index length:", len(index))
if len(index) != 906290:
    print("Warning: the length of the index looks wrong.")
    print("Make sure you are using `process_tokens_original` when you build the index.")
    raise Exception()

# the list of queries asked for in the assignment text
queries = [
    "Welcoming",
    "Australasia OR logistic",
    "heart AND warm",
    "global AND space AND wildlife",
    "engine OR origin AND record AND wireless",
    "placement AND sensor OR max AND speed"
]

# run each of the queries and print the result
ids_to_doc = {v:k for k, v in doc_ids.items()}
for q in queries:
    res = run_boolean_query(q, index)
    res.sort(key=lambda x: ids_to_doc[x])
    print(q)
    for r in res:
        print(ids_to_doc[r])

