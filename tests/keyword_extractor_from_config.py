from keyword_extraction.helpers import init_keyword_extractor

if __name__ == '__main__':
    config = {
        "name": "EmbedRankTransformer1",
        "class": "KeywordExtractor",
        "encoder": {
            "class": "BertClient",
            "kwargs": {
            }
        },
        "nlp": {
            "name": "spacy",
            "model_name": "en_core_web_sm"
        },
        "extractor": {
            "class": "PhraseExtractor",
            "kwargs": {
                "grammar": "GRAMMAR1",
                "np_method": "NOUN_CHUNKS",
                "np_tags": "NLTK",
                "stopwords": "NLTK"
            }
        },
        "embedding": {
            "class": "NaiveEmbedding",
            "kwargs": {}
        },
        "rank": {
            "class": "EmbedRank",
            "kwargs": {
                "emb_method": "NAIVE",
                "mmr_beta": 0.55,
                "top_n": 10,
                "alias_threshold": 0.8
            }
        }
    }

    keyword_extractor = init_keyword_extractor(config)

    text = "NuVox shows staying power with new cash, new market Who says you can't raise cash in today's telecom " \
           "market? NuVox Communications positions itself for the long run with $78.5 million in funding and a " \
           "new credit facility"

    phrases, relevance = keyword_extractor.run(text)

    print(phrases, relevance)
