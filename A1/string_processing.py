from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


def process_tokens(toks):
    # TODO: fill in the functions: process_tokens_1, 
    # process_tokens_2, and process_tokens_3 functions and
    # uncomment the one you want to test below

    # NOTE: make sure to switch back to process_tokens_original
    # and rebuild the index before
    # tackling the other assignment questions

    #return process_tokens_1(toks)
    #return process_tokens_2(toks)
    #return process_tokens_3(toks)
    return process_tokens_original(toks)

# get the nltk stopwords list
stopwords = set(nltk.corpus.stopwords.words("english"))
def process_tokens_original(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        new_toks.append(t)
    return new_toks

def process_tokens_1(toks): #This is 
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    stemmer = PorterStemmer()
    lem = WordNetLemmatizer()
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords or not t.isalpha() or not t.lower().isalpha(): #start by removing punctuation
            continue
        #TODO: your code should modify t and/or do some sort of filtering
        t = lem.lemmatize(stemmer.stem(t))
        new_toks.append(t) #Normalize by removing punctuation and lower casing the words
    #print('Old: '+str(len(toks)), 'Updated: '+str(len(new_toks)), 'Delta: ' + str(len(toks)-len(new_toks))) #Part of the performance metrics since we want to consider how we can reduce the scope of the problem
    return new_toks

def process_tokens_2(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """

    new_toks = []
    stemmer = SnowballStemmer("english")
    lem = WordNetLemmatizer()
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords or not t.isalpha() or not t.lower().isalpha(): #start by removing punctuation
            continue
        #TODO: your code should modify t and/or do some sort of filtering
        t = lem.lemmatize(stemmer.stem(t))
        new_toks.append(t)
    return new_toks

def process_tokens_3(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    stemmer = LancasterStemmer()
    lem = WordNetLemmatizer()
    for t in toks:
       # ignore stopwords
        if t in stopwords or t.lower() in stopwords or not t.isalpha() or not t.lower().isalpha(): #start by removing punctuation
            continue
        t = lem.lemmatize(stemmer.stem(t)) #Same as experiment one but we normalize everything to lowercase. This is has been mentioned in the lectures as a bad idea, however thought it would be good to see the impact on eval
        new_toks.append(t)
    return new_toks




def tokenize_text(data):
    """Convert a document as a string into a document as a list of
    tokens. The tokens are strings.

    Args:
        data (str): The input document

    Returns:
        list(str): The list of tokens.
    """
    # split text on spaces
    tokens = data.split()
    return tokens

