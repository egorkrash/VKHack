import pandas as pd
import re
import pymorphy2
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


df = pd.read_csv('out.csv')
df = df[(df['message'].isnull() == False)
        & (df['message'].isin(['Перевод из приложения СНГБ Онлайн', 'Перевод', 'Тест']) == False)]

df['message'] = df['message'].apply(lambda x: x.replace('Перевод из приложения СНГБ Онлайн:', ''))
df['message'] = df['message'].apply(lambda x: x.replace('!', ' вз'))
df['flag'] = df['message'].apply(lambda x: 1 if 'RaiffeisenC2C' in x else 0)
df = df[df['flag'] != 1]

df.drop('flag', axis=1, inplace=True)
for i in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
    df['message'] = df['message'].apply(lambda x: x.replace(i, ''))

df = df[df['message'] != '']

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
        return filter(lambda x: len(x) > 2, re.findall(r'(?u)\w+', text))
    else:
        return re.findall(r'(?u)\w+', text)


def text2canonicals(text, add_word=False, filter_short_words=True):
    words = []
    for word in get_words(text, filter_short_words=filter_short_words):
        words.append(word2canonical(word.lower()))
        if add_word:
            words.append(word.lower())
    return words


result = parallelization(text2canonicals, df['message'].values)
df['lemma'] = result
df['len_lemma'] = df['lemma'].apply(lambda x: len(x))
df = df[df['len_lemma'] != 0]

df.drop('len_lemma', axis=1).to_pickle('pickles/new_data.pkl')
