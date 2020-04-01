import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import roc_curve, auc


def clean_text(text: str) -> str:
    """ remove unwanted characters from text """
    # remove line breaks
    text = text.replace('\n', '')
    # remove quotes
    text = text.replace('"', '')
    #lowercase text
    text = text.lower()
    #remove punctuation and special characters ?:!.,[]();
    text = text.translate(str.maketrans('', '', '?:!.,;[]()*-'))
    # remove possessive pronouns termination
    text = text.replace("'s", "")
    # remove unnecessery white spaces
    text = re.sub(r"\s\s+", " ", text)
        
    return text

def apply_cv(cls, x_train, y_train, metric, cv=3):
    cv_results = cross_validate(cls, x_train, y_train, scoring=metric, return_train_score=True, cv=cv)
    print(f"CV {metric}: {round(cv_results['test_score'].mean() * 100, 2)}%")

def perform_grid_search(cls, param_grid, x_train, y_train, metric):
    """ perform a grid search on a classifier using 3-fold cross validation and the passed prameters """
    grid_search = GridSearchCV(estimator=cls, 
                               param_grid=param_grid,
                               scoring=metric,
                               cv=3,
                               verbose=1)
    grid_search.fit(x_train, y_train)
    print(f"The best hyperparameters from Grid Seach are: {grid_search.best_params_}")
    return grid_search.best_estimator_

def clean2(raw_text, lemmatizer, stemmer, stop_words = []):
    # remove special characters such as the "'" in "it's".
    text = re.sub(r'\W', ' ', raw_text)
    
    # remove digits
    text = re.sub("\d+", "", text)

    # remove single character such as the "s" in "it s".
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # remove stop words if it's not none.
    for w in stop_words:
        text = re.sub(r"\b" + w + r"\b", '', text)

    # unify successive blank space as one blank space.
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    
    text = text.split(' ')
    # lemmatize ('went' -> 'go') and stem ('photographers' -> 'photographer') each word in the text.
    stemmed_text = []
    for w in text:
        if w != '':
            lemmatized_word = lemmatizer.lemmatize(w, pos='v')
            stemmed_word = stemmer.stem(lemmatized_word)
            stemmed_text.append(stemmed_word)

    text = ' '.join(stemmed_text)
    
    return text

def remove_ambigous_sentences(df: pd.DataFrame, col: str) -> pd.DataFrame:
    for positive_senten in df.loc[df[col]==1, 'text']:
        df.drop(df[(df[col]==0) & (df['text']==positive_senten)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def find_opti_thresh(y, proba, plot = True):
    ###
    # Given true labels y_test and predicted probabilities y_proba, determine the auroc score
    # and find the threshold that maximize the recall for both classes
    ###

    # calculate auroc score roc_auc
    fpr, tpr, thresholds = roc_curve(y, proba)
    roc_auc = auc(fpr,tpr)

    # find the threshold that maximize recall of both classes specified with condition "s = 1 - fpr[i] + tpr[i]"
    maxi = 0
    max_ind = 0
    for i in range(len(fpr)):
        s = 1 - fpr[i] + tpr[i]
        if s > maxi:
            maxi = s
            max_ind = i

    best_thresh = thresholds[max_ind]

    # plot the ROC curve 
    if plot == True:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot([fpr[max_ind]], [tpr[max_ind]], marker='o', markersize=10, color="red")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - recall of 0')
        plt.ylabel('recall of 1')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
        print(f'optimal threshold is {best_thresh}')

    return best_thresh, roc_auc