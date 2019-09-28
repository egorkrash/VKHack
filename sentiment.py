import pandas as pd
import re
import pymorphy2
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle as pkl
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
        
tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)

# morph analyzer for text lemmatization
morph = pymorphy2.MorphAnalyzer()
fasttext = FastTextKeyedVectors.load('187/model.model')
pos_log_reg = pkl.load(open('pos_log_reg.pkl', 'rb'))
neg_log_reg = pkl.load(open('neg_log_reg.pkl', 'rb'))
pos_log_reg_dost = pkl.load(open('pos_log_reg_dost.pkl', 'rb'))
neg_log_reg_dost = pkl.load(open('neg_log_reg_dost.pkl', 'rb'))

# function for performing parallel computing on cpu
def parallelization(func, massive, jobs=None, tq=True):
    num_cores = multiprocessing.cpu_count() if jobs is None else jobs
    if tq:
        results = np.array(Parallel(n_jobs=num_cores)(delayed(func)(i) for i in tqdm(massive)))
        return results
    else:
        results = Parallel(n_jobs=num_cores)(delayed(func)(i) for i in massive)
        return results


def _word2canonical4w2v(word):
    elems = morph.parse(word)
    my_tag = ''
    res = []
    for elem in elems:
        if 'VERB' in elem.tag or 'GRND' in elem.tag or 'INFN' in elem.tag:
            my_tag = 'V'
        if 'NOUN' in elem.tag:
            my_tag = 'S'
        normalised = elem.normalized.word
        res.append((normalised, my_tag))
    tmp = list(filter(lambda x: x[1] != '', res))
    if len(tmp) > 0:
        return tmp[0]
    else:
        return res[0]


def word2canonical(word):
    return _word2canonical4w2v(word)[0]


def get_words(text, filter_short_words=False):
    if filter_short_words:
        return filter(lambda x: len(x) > 2, re.findall('[а-яА-Яa-zA-Z]+', text))#re.findall(r'(?u)\w+', text))
    else:
        return re.findall(r'(?u)\w+', text)

def text2canonicals(text, add_word=False, filter_short_words=True):
    words = []
    for word in get_words(text, filter_short_words=filter_short_words):
        words.append(word2canonical(word.lower()))
        if add_word:
            words.append(word.lower())
    return words


def get_text_vectors(text):
    matrix = np.zeros((len(text), 300))
    for i,word in enumerate(text):
        vector = fasttext[word]
        matrix[i] = vector
        
    return matrix

def get_dost_vector(pred):
    return np.array([pred['positive'], pred['skip'], pred['speech'], pred['neutral'], pred['positive']])


def preprocess(texts):
    # embedding vectors weighted with tfidf
    preprocessed_texts = parallelization(text2canonicals, texts)
    lengths = np.array(list(map(lambda x: len(x) if len(x) > 0 else 1, preprocessed_texts)))
    
    texts = list(map(lambda x: ' '.join(x), preprocessed_texts))
    vectorizer = TfidfVectorizer()
    tfifd_vectorized = vectorizer.fit_transform(texts).toarray()
    unique_words = list(map(lambda x: x[0], sorted(vectorizer.vocabulary_.items())))
    
    all_vectors = get_text_vectors(unique_words)
    weighted_embeddings = tfifd_vectorized @ all_vectors
    weighted_embeddings /= lengths.reshape(-1, 1)
    del tfifd_vectorized, all_vectors
    
    return weighted_embeddings


def preprocess2(texts, use_dost=False):
    # mean embedding vectors
    if use_dost:
        preds = model.predict(texts)
        dost_vectors = np.array(list(map(get_dost_vector, preds)))
    
    #preprocessed_texts = parallelization(text2canonicals, texts)
    preprocessed_texts = list(map(lambda x: x.lower().split(), texts))
    
    embeddings = np.zeros((len(texts), 300))
    for i, text in enumerate(preprocessed_texts):
        vectors = get_text_vectors(text)
        if vectors.shape[0] > 0:
            vector = np.mean(vectors, axis=0)
        else:
            vector = np.random.randn(300,)
        embeddings[i] = vector
        
    if use_dost:
        return np.concatenate((embeddings, dost_vectors), axis=1)
    return embeddings

def emotional(x):
    emotions = {')', '!', '('}
    for e in emotions:
        if e in x:
            return 1
    return 0


def sentiment_analysis(sentences, use_dost=True):
    emotional_col = list(map(lambda x: emotional(x), sentences))
    prep_sent = preprocess2(sentences, True)
    inputs = np.concatenate((prep_sent, np.array(emotional_col).reshape(-1, 1)), axis=1)
    if use_dost:
        preds_pos = list(map(lambda x: np.round(x[1], 3), pos_log_reg_dost.predict_proba(inputs)))
        preds_neg = list(map(lambda x: np.round(x[1], 3), neg_log_reg_dost.predict_proba(inputs)))
    else:
        preds_pos = list(map(lambda x: np.round(x[1], 3), pos_log_reg.predict_proba(inputs)))
        preds_neg = list(map(lambda x: np.round(x[1], 3), neg_log_reg.predict_proba(inputs)))
    
    return np.array([preds_pos, preds_neg]).T
