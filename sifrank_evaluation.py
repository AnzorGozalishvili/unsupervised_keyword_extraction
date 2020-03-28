#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/25
import json
import time
import tqdm

import nltk
import pandas as pd
import requests

from helpers import read_json
from keyword_extraction.helpers import init_keyword_extractor
from utils import fileIO


def calculate_scores(num_c, num_e, num_s):
    P = float(num_c) / float(num_e)
    R = float(num_c) / float(num_s)
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


def scores_to_dict(P, R, F1, N):
    return {
        f"P.{N}": P,
        f"R.{N}": R,
        f"F1.{N}": F1
    }


def generate_scores_table(model_names, model_scores):
    table = pd.DataFrame(model_scores, index=model_names)
    table.index.name = "Models"
    return table


def save_scores(scores, path, format='csv'):
    with open(path, 'w') as file:
        if format == "csv":
            for dataset_name, scores_table in scores.items():
                file.write(f"Evaluation results on \*\*{dataset_name}\*\*" + "\n")
                file.write(scores_table.to_csv() + "\n")
        elif format == 'json':
            scores_dict = {
                dataset_name: scores.to_dict(orient='index') for dataset_name, scores in scores.items()
            }
            json.dump(scores_dict, file)


class EmbedRankWrapper:
    def __init__(self, url=None):
        self.url = url or "http://0.0.0.0:5000/"

    def run(self, text, top_n=15):
        data = {"text": text, "n": top_n}
        result = requests.post(self.url, json=data)
        content = json.loads(result.content)
        keywords = [(keyword, score) for keyword, score in zip(content[0], content[1])]

        return keywords


class EmbedRankTransformersWrapper:
    def __init__(self, config_path=None):
        self.conf_path = config_path or 'evaluation/config/embedrank_bert_as_a_service.json'
        self.model = init_keyword_extractor(read_json(self.conf_path))

    def run(self, text, top_n=15):
        keywords, relevance = self.model.run(text, )
        keywords = [(keyword, score) for (keyword, _, _), score in zip(keywords, relevance)]

        return keywords


class SIFRankWrapper:
    def __init__(self, url=None):
        self.url = url or "http://0.0.0.0:5001/"

    def run(self, text, top_n=15):
        data = {"text": text, "n": top_n}
        result = requests.post(self.url, json=data)
        content = json.loads(result.content)
        keywords = [(keyword, score) for keyword, score in zip(content[0], content[1])]

        return keywords


def get_model(model_name, **kwargs):
    if model_name == 'EmbedRank':
        return EmbedRankWrapper(**kwargs)
    elif 'EmbedRank' in model_name:
        return EmbedRankTransformersWrapper(**kwargs)
    elif model_name == 'SIFRank' or 'SIFRankPlus':
        return SIFRankWrapper(**kwargs)


def get_dataset(dataset_name):
    if dataset_name == "Inspec":
        data, labels = fileIO.get_inspec_data()
        lamda = 0.6
        elmo_layers_weight = [0.0, 1.0, 0.0]
    elif dataset_name == "Duc2001":
        data, labels = fileIO.get_duc2001_data()
        lamda = 1.0
        elmo_layers_weight = [1.0, 0.0, 0.0]
    else:
        data, labels = fileIO.get_semeval2017_data()
        lamda = 0.6
        elmo_layers_weight = [1.0, 0.0, 0.0]

    return data, labels, lamda, elmo_layers_weight


def evaluate(model_name, dataset_name, model_kwargs):
    time_start = time.time()

    P = R = F1 = 0.0
    num_c_5 = num_c_10 = num_c_15 = 0
    num_e_5 = num_e_10 = num_e_15 = 0
    num_s = 0
    lamda = 0.0

    data, labels, lamda, elmo_layers_weight = get_dataset(dataset_name)

    porter = nltk.PorterStemmer()  # please download nltk

    model = get_model(model_name, **model_kwargs)
    print(f"successfully loaded {model_name} model with params {model_kwargs}")

    for key, data in tqdm.tqdm(data.items(), desc=f"Run {model_name} on {dataset_name} records..."):

        lables = labels[key]
        lables_stemed = []

        for lable in lables:
            tokens = lable.split()
            lables_stemed.append(' '.join(porter.stem(t) for t in tokens))

        keywords = model.run(data)

        j = 0
        for temp in keywords[0:15]:
            tokens = temp[0].split()
            tt = ' '.join(porter.stem(t) for t in tokens)
            if (tt in lables_stemed or temp[0] in labels[key]):
                if (j < 5):
                    num_c_5 += 1
                    num_c_10 += 1
                    num_c_15 += 1

                elif (j < 10 and j >= 5):
                    num_c_10 += 1
                    num_c_15 += 1

                elif (j < 15 and j >= 10):
                    num_c_15 += 1
            j += 1

        if (len(keywords[0:5]) == 5):
            num_e_5 += 5
        else:
            num_e_5 += len(keywords[0:5])

        if (len(keywords[0:10]) == 10):
            num_e_10 += 10
        else:
            num_e_10 += len(keywords[0:10])

        if (len(keywords[0:15]) == 15):
            num_e_15 += 15
        else:
            num_e_15 += len(keywords[0:15])

        num_s += len(labels[key])

    p, r, f = calculate_scores(num_c_5, num_e_5, num_s)
    scores_5 = scores_to_dict(p, r, f, 5)
    p, r, f = calculate_scores(num_c_10, num_e_10, num_s)
    scores_10 = scores_to_dict(p, r, f, 10)
    p, r, f = calculate_scores(num_c_15, num_e_15, num_s)
    scores_15 = scores_to_dict(p, r, f, 15)

    scores = {
        **scores_5,
        **scores_10,
        **scores_15,
        "time": time.time() - time_start
    }

    return scores


if __name__ == '__main__':
    model_names = [
        "EmbedRankBERT",
        "EmbedRankSentenceBERT",
        "EmbedRank",
        "SIFRank",
        "SIFRankPlus"
    ]

    model_params = [
        {"config_path": 'evaluation/config/embedrank_bert_as_a_service.json'},
        {"config_path": 'evaluation/config/embedrank_sentence_bert.json'},
        {"url": "http://0.0.0.0:5000"},
        {"url": "http://0.0.0.0:5001/sifrank"},
        {"url": "http://0.0.0.0:5001/sifrankplus"}
    ]

    dataset_names = [
        "Inspec",
        #        "Duc2001",
        "Semeval2017"
    ]

    scores_path = "sifrank_eval_results.csv"
    scores_format = "csv"

    scores = {}

    for dataset_name in dataset_names:
        evaluated_models = []
        evaluated_model_scores = []

        for model_name, params in zip(model_names, model_params):
            model_scores = evaluate(model_name, dataset_name, params)
            evaluated_model_scores.append(model_scores)
            evaluated_models.append(model_name)

        scores_table = generate_scores_table(evaluated_models, evaluated_model_scores)

        scores[dataset_name] = scores_table

    save_scores(scores, scores_path, scores_format)
