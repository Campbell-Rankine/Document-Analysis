import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from tqdm import tqdm
from nltk.tokenize import TreebankWordTokenizer

#optimization imports
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w:v for w, v in zip(data["words"], data["vectors"])}

tokenizer = TreebankWordTokenizer()
# convert a document into a vector
def document_to_vector(doc):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensionals.
    """
    # TODO: tokenize the input document
    toks = tokenizer.tokenize(doc)
    toks = [x for x in toks if x in w2v.keys()]
    # TODO: aggregate the vectors of words in the input document
    vec = np.zeros(300)
    for tok in toks:
        vec += w2v[tok]
    return vec* (1/len(toks))
            

# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the 
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    #TODO: convert each of the training documents into a vector
    X = [document_to_vector(x) for x in tqdm(Xtr)] #Convert to vector list
    #TODO: train the logistic regression classifier
    model = LogisticRegression(C=C)
    print("Training")

    model.fit(X, Ytr)
    return model

# fit a linear model 
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    #TODO: convert each of the testing documents into a vector
    XTes = [document_to_vector(x) for x in Xtst] #Convert to vector list
    #TODO: test the logistic regression classifier and calculate the accuracy
    score = accuracy_score(Ytst, model.predict(XTes))
    return score



def main():
    print("Starting")
    print("find base params...")
    X = [document_to_vector(x) for x in tqdm(X_train)]
    print("train base...")
    model = fit_model(X_train, Y_train, 1.0)
    score = test_model(model, X_val, Y_val)
    print("Score (C= 1.0): " + str(score))

    args = (X, Y_train)
    grid = dict(C= [1.0, 2.5, 5.0, 10.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 100.0])
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    model = LogisticRegression()
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(*args) #run optimization
    score = grid_result.best_score_
    valC = grid_result.best_params_
    print(score, valC)

    print("retest w our structure")
    model = fit_model(X_train, Y_train, valC['C'])
    score1 = test_model(model, X_val, Y_val)
    print("Score: " + str(test_model(model, X_test, Y_test)) + " Diff: " + str(score1 - score))

    print("Rerun with concatenated set")
    Xtr = X_train + X_val
    Ytr = Y_train + Y_val

    X = [document_to_vector(x) for x in tqdm(Xtr)]
    print("train base...")
    model = fit_model(Xtr, Ytr, 1.0)
    score = test_model(model, X_test, Y_test)
    print("Score (C= 1.0): " + str(score))

    args = (X, Ytr)
    grid = dict(C= [1.0, 2.5, 5.0, 10.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 100.0])
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    model = LogisticRegression()
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(*args) #run optimization
    score = grid_result.best_score_
    valC = grid_result.best_params_
    print(score, valC)

    print("retest w our structure")
    model = fit_model(Xtr, Ytr, valC['C'])
    score1 = test_model(model, X_test, Y_test)
    print("Score: " + str(score1) + " Diff: " + str(score1 - score))

# TODO: search for the best C parameter using the validation set
### - BAYESOPT - ###
#def objective(C):
#    #Define experiment space
#    model = fit_model(X_train, Y_train, C)
 #   return -test_model(model, X_test, Y_test)


# TODO: fit the model to the concatenated training and validation set
#   test on the test set and print the result

if __name__ == '__main__':
    main()
