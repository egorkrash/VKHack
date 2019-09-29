import multiprocessing
import pickle
import re
from functools import lru_cache
from time import time
import numpy as np
import pandas as pd
import pymorphy2
from gensim.models import KeyedVectors
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.test.utils import datapath
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

np.random.seed(42)

clusters_names = {
0: "Обед",
1: "Оплата услуг",
2: "Сентябрь",
3: "Такси",
4: "Спасибо",
5: "ФИО",
6: "Долг",
7: "Перевод",
8: "Проверка связи",
9: "Тебе",
10: "Центральный",
11: "иммя",
12: "Деньги",
13: "ФИО",
14: "Привет",
15: "Центральный",
16: "Центральный",
17: "Тест",
18: "ФИО",
19: "Подарок",
20: "ФИО",
21: "Перевод",
22: "Лови",
23: "ФИО",
24: "Мама",
25: "Себе",
26: "Тест",
27: "Проверка связи",
28: "Проба",
29: "ФИО",
30: "Проверка связи",
31: "За сигареты",
32: "English",
33: "Центральный",
34: "ФИО",
35: "Центральный",
36: "Тест",
37: "Проверка связи",
38: "Цветы",
39: "День рождения",
40: "Оплата",
41: "ФИО",
42: "Спасибо",
43: "English",
44: "Рахмат",
45: "Оплата",
46: "Страховка",
47: "Центральный",
48: "За квартиру",
49: "ФИО"
}

# Start service with these models in memory
w2v_vectors = KeyedVectors.load_word2vec_format("185/model.bin", binary=True)
w2v_vectors.vocab = dict(zip(list(map(lambda x: x.split('_')[0], w2v_vectors.vocab.keys())), w2v_vectors.vocab.values()))
kmeans = pickle.load(open('pickles/kmeans.pkl', 'rb'))
comments = pickle.load(open('pickles/comments.pkl', 'rb'))
# morph analyzer for text lemmatization
morph = pymorphy2.MorphAnalyzer()


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
        return filter(lambda x: len(x) > 3, re.findall(r'(?u)\w+', text))
    else:
        return re.findall(r'(?u)\w+', text)


def text2canonicals(text, add_word=False, filter_short_words=True):
    words = []
    for word in get_words(text, filter_short_words=filter_short_words):
        words.append(word2canonical(word.lower()))
        if add_word:
            words.append(word.lower())
    return words


def preprocess(texts, dump=True):
    preprocessed_texts = parallelization(text2canonicals, texts)
    vectorizer = TfidfVectorizer()
    texts = list(map(lambda x: ' '.join(x), preprocessed_texts))

    if dump:
        vectorizer = vectorizer.fit(texts)
        with open('pickles/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    else:
        with open('pickles/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

    tfifd_vectorized = vectorizer.transform(texts).toarray()
    unique_words = list(map(lambda x: x[0], sorted(vectorizer.vocabulary_.items())))

    all_vectors = get_text_vectors(unique_words)
    weighted_embeddings = tfifd_vectorized @ all_vectors

    del tfifd_vectorized, all_vectors
    
    return weighted_embeddings

def preprocess_single_text(text):
    # embedding vectors weighted with tfidf
    preprocessed_text = text2canonicals(text)
    length = len(preprocessed_text) if len(preprocessed_text) > 0 else 1
    
    preprocessed_text = ' '.join(preprocessed_text)
    vectorizer = pickle.load(open('pickles/vectorizer.pkl', 'rb'))
    tfifd_vectorized = vectorizer.transform([preprocessed_text]).toarray()
    unique_words = list(map(lambda x: x[0], sorted(vectorizer.vocabulary_.items())))
    
    all_vectors = get_text_vectors(unique_words)
    weighted_embeddings = tfifd_vectorized @ all_vectors
    weighted_embeddings /= length
    del tfifd_vectorized, all_vectors
    
    return weighted_embeddings

def get_text_vectors(text):
    cnt = 0
    matrix = np.zeros((len(text), 300))
    for i,word in enumerate(text):
        try:
            vector = w2v_vectors[word]
        except KeyError:
            cnt += 1
            vector = np.zeros((300,))
        matrix[i] = vector
    #print('cached {} exeptions'.format(cnt))
    return matrix


def infer_cluster(text):
    vector = preprocess_single_text(text)
    cluster = kmeans.predict(vector)[0]
    try:
        random_3 = comments.query('cluster == @cluster').sample(3, random_state=0)['content'].to_list()
    except ValueError:
        random_3 = comments.query('cluster == @cluster').sample(3, random_state=0, replace=True).to_list()
    return clusters_names[cluster], random_3


def get_top_clusters_overall(comments=comments):
    cluster_frequencies = (comments["cluster"]
                           .value_counts()[:3]
                           .map(lambda x: str(round(x/len(comments) * 100, 2))+"%")
                           .to_dict()
                           )
    return {clusters_names[k]: v for k, v in cluster_frequencies.items()}


def get_top_clusters_month(comments=comments):
    comments = comments[(comments["local_datetime"].dt.month) == 7 & (comments["local_datetime"].dt.year == 2019)]
    cluster_frequencies = (comments["cluster"]
                           .value_counts()[:3]
                           .map(lambda x: str(round(x/len(comments) * 100, 2))+"%")
                           .to_dict()
                           )
    return {clusters_names[k]: v for k, v in cluster_frequencies.items()}


def get_top_clusters_week(comments=comments):
    comments = comments[("2019-07-25" <= comments["local_datetime"]) & (comments["local_datetime"] <= "2019-07-31")]
    cluster_frequencies = (comments["cluster"]
                           .value_counts()[:3]
                           .map(lambda x: str(round(x/len(comments) * 100, 2))+"%")
                           .to_dict()
                           )
    return {clusters_names[k]: v for k, v in cluster_frequencies.items()}

