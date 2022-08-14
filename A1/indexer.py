import numpy as np
import glob
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy import true

from string_processing import *


def read_doc(file_path):
    """Read a document from a path, tokenize, process it and return
    the list of tokens.

    Args:
        file_path (str): path to document file

    Returns:
        list(str): list of processed tokens
    """
    data = open(file_path, "r", encoding='utf-8').read()
    toks = tokenize_text(data)
    toks = process_tokens(toks)
    return toks

def gov_list_docs(docs_path):
    """List the documents in the gov directory.
    Makes explicit use of the gov directory structure and is not
    a general solution for finding documents.

    Args:
        docs_path (str): path to the gov directory

    Returns:
        list(str): list of paths to the document 
    """
    path_list = []
    # get all directories in gov root folder
    dirs = glob.glob(os.path.join(docs_path, "*"))
    for d in dirs:
        # get all the files in each of the sub directories
        files = glob.glob(os.path.join(d, "*"))
        path_list.extend(files)
    return path_list

def make_doc_ids(path_list):
    """Assign unique doc_ids to documents.

    Args:
        path_list (list(str)): list of document paths 

    Returns:
        dict(str : int): dictionary of document paths to document ids
    """
    cur_docid = 0
    doc_ids = {}
    for p in path_list:
        # assign docid
        doc_ids[p] = cur_docid
        # increase docid
        cur_docid += 1
    return doc_ids

def get_token_list(path_list, doc_ids):
    """Read all the documents and get a list of all the tokens

    Args:
        path_list (list(str)): list of paths
        doc_ids (dict(str : int)): dictionary mapping a path to a doc_id

    Returns:
        list(tuple(str, int)): an asc sorted list of token, doc_id tuples
    """
    all_toks = []
    for path in tqdm(path_list):
        doc_id = doc_ids[path]
        toks = read_doc(path)
        for tok in toks:
            all_toks.append((tok, doc_id))
    return sorted(all_toks)

def index_from_tokens(all_toks):
    """Construct an index from the sorted list of token, doc_id tuples.

    Args:
        all_toks (list(tuple(str, int))): an asc sorted list of (token, doc_id) tuples
            this is sorted first by token, then by doc_id

    Returns:
        tuple(dict(str: list(tuple(int, int))), dict(str : int)): a dictionary that maps tokens to
        list of doc_id, term frequency tuples. Also a dictionary that maps tokens to document 
        frequency.
    """

    # TODO: implement this function.
    #initialize doc frequency
    doc_freq = dict()
    index = dict()
    for tok in tqdm(all_toks):
        if not tok[0] in index.keys():
            #Add key to both dictionaries
            index[tok[0]] = []
        #if key is in update
        contain = False
        for i, item in enumerate(index[tok[0]]):
            #Check whether or not Doc_ID matches, if so update frequency
            if item[0] == tok[1]: #If doc ID's match, then add 1 to the 
                contain = True
                index[tok[0]][i] = (item[0], item[1] + 1) #update frequency
        if not contain:
            index[tok[0]].append((tok[1], 1))
    for x in index.keys():
        doc_freq[x] = len(index[x])
        index[x] = sorted(index[x], key=lambda tup: tup[0]) #enforce sort
    return index, doc_freq

# run the index example given in the assignment text
print(index_from_tokens([("cat", 1), ("cat", 1), ("cat", 2), ("door", 1), ("water", 3)]))

# get a list of documents 
doc_list = gov_list_docs("./gov/documents")
print("Found %d documents." % len(doc_list))
num_docs = len(doc_list)

# assign unique doc_ids to each of the documents
doc_ids = make_doc_ids(doc_list)
ids_to_doc = {v:k for k, v in doc_ids.items()}

# get the list of tokens in all the documents
tok_list = get_token_list(doc_list, doc_ids)

print("indexing: ")
# build the index from the list of tokens
index, doc_freq = index_from_tokens(tok_list)
del tok_list # free some memory

# store the index to disk
pickle.dump((index, doc_freq, doc_ids, num_docs), open("stored_index.pik", "wb"))

