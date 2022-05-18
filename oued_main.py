import pandas as pd
import os
from threading import Thread
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import regex
import nltk
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from threading import Thread
import math
import unidecode
from params import *
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import time
import numpy as np
from oued_timing import Timer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from pathlib import Path


#nltk.downlod('all')
class Preprocessor:
    def __init__(self):
        self.lemmatizer = FrenchLefffLemmatizer()
        self.french_stopwords = set(stopwords.words('french'))
        self.stemmer= nltk.stem.snowball.FrenchStemmer()

    def get_int_array(self, str_table):
        ok = False
        value = 0
        ans = []
        for c in str_table:
            if not c.isnumeric():
                if ok:
                    ans.append(value)
                    value = 0
                ok = False
            else:
                ok = True
                value *= 10
                value += int(c)
        assert ans
        return ans
    def get_int_array_min(self, str_table):
        return min(self.get_int_array(str_table))
    def clean_txt(self, text):
        filtre_stopfr =  lambda text: [token for token in text if token not in self.french_stopwords]
        text = str(text)
        text = text.lower()
        text = re.sub("""[\.\!\"\s\?\-\,\'@&^$`]+""", " ",text)
        text = re.sub("""(\W|\d)+""", " ",text)
        text = regex.sub(u'[^\p{Latin}]', u' ', text)
        words = filtre_stopfr(word_tokenize(text, language="french") )
        words = [self.lemmatizer.lemmatize(unidecode.unidecode(w)) for w in words]
        words = [w for w in words if 3 < len(w) and len(w) < 13]
        #words = [self.stemmer.stem(w) for w in words]
        return " ".join(words)


def all(src_file_path, dest_file_path): 
    df = pd.read_csv(src_file_path); 
    pre = Preprocessor()
    def handle(df, start, end):
        print(f"start {start} {end}")
        end = min(end, df.shape[0])
        df.loc[start:end, "title"] = df.loc[start:end,"title"].apply(pre.clean_txt)
        df.loc[start:end, "description"] = df.loc[start:end,"description"].apply(pre.clean_txt)
        print(f"end {start} {end}")
    
    rows = df.shape[0]
    step = math.ceil(rows / NBR_THREADS)
    threads = []
    df["root"] = df["categories"].apply(pre.get_int_array_min)
    for i in range(0, rows, step):
        threads.append(Thread(target=handle, args=(df, i, min(rows, i + step))))
        threads[-1].start()
    for t in threads:
        t.join()
    df["title"] = df["title"].fillna(" ")
    df["description"] = df["description"].fillna(" ")
    df["text"] = df["title"] + " " + df["description"]
    df["text"] = df["text"].fillna(" ")
    df.to_csv(dest_file_path);

def learn_text(serie):
    vectorizer = TfidfVectorizer()
    oued = vectorizer.fit_transform((serie))
    return oued

def learn_category(df, category):
    path = Path(f"{DATA_FRAMES_DIR}/{category}")
    path.mkdir(parents=True, exist_ok=True)

    cur = df[df["root"] == category];

    df_text = cur[["id", "title", "root", "description", "status"]]
    tfidf_title = learn_text(df_text['title'])
    tfidf_description = learn_text(df_text['description'])
    tfidf_text = learn_text(df_text['text'])
    dump(tfidf_title, path / "title_model.sav")
    dump(tfidf_description, path / "description_model.sav")
    dump(tfidf_text, path / "text_model.sav")
    df_text.to_csv(path / "text_preprocess.csv");


def get_roots(df):
    return list(map(int, df["root"].unique()))

class CategoryController:
    def __init__(self, df, title, description):
        self.description = description[(df["status"] == 1)|(df["status"] == 4)]
        self.title = title[(df["status"] == 1)|(df["status"] == 4)]
        cur = df[(df["status"] == 1)|(df["status"] == 4)]
        cur["idx"] = range(cur.shape[0])
        cur = cur[["id", "idx"]]
        self.ids = cur["idx"]
        self.ids.index = cur["id"]

    @Timer(name="get_score")
    def get_score(self, ann_id):
        index = self.ids.get(ann_id, default=-1)
        if index < 0:
            return []
        title_values = cosine_similarity(self.title, self.title[index]).flatten()
        description_values = cosine_similarity(self.description, self.description[index]).flatten()
        values = (title_values * TITLE_ALPHA) + (description_values * (1 - TITLE_ALPHA))
        torem = values <= THRESHOLD
        ans = zip(np.delete(values, torem), np.delete(self.ids, torem))
        return ans


class Controller:
    def __init__(self, roots):
        self.roots = roots;
        self.M = dict()
        for root in self.roots:
            path = Path(f"{DATA_FRAMES_DIR}/{root}")
            df = pd.read_csv(path / "text_preprocess.csv")
            title = load(path / "title_model.sav")
            description = load(path / "description_model.sav")
            self.M[root] = CategoryController(df, title, description)

    def recommend(self, root, ann_id):
        if root not in self.M:
            return []
        return self.M[root].get_score(ann_id)
        
    
def recommend(ids, title_model, description_model, index, count):
    title_values = cosine_similarity(title_model,title_model[index])
    description_values = cosine_similarity(description_model,description_model[index])
    values = (tile_values * TITLE_ALPHA) + (description_values * (1 - TITLE_ALPHA))
    ans = sorted(zip(values, ids), key=lambda p: p[0], reverse=True)
    return ans[:count]

def learn(src_db):
    df = pd.read_csv(src_db)
    df["title"] = df["title"].fillna(" ")
    df["description"] = df["description"].fillna(" ")
    df["text"] = df["text"].fillna(" ")
    folders = get_roots(df)
    for folder in folders:
        learn_category(df, folder)
        print(f"{folder} learned")

#all("../ann.csv", "./result.csv")

#cnt = oued_main.Controller(roots)
#ls = cnt.recommend(239, 946056)
