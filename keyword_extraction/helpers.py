import bert_serving.client as encoders
import spacy
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage

import embedding.embedding as embeddings
import extraction.extractor as extractors
import keyword_extraction.keyword_extractor as keyword_extractors
import rank.rank as ranks


class NLPs:
    SPACY = 'spacy'
    CORENLP = 'corenlp'


def init_nlp(config):
    if config.get('name') == NLPs.SPACY:
        return spacy.load(config.get('model_name'))
    elif config.get('name') == NLPs.CORENLP:
        return StanfordNLPLanguage(stanfordnlp.Pipeline(lang=config.get('model_name')))


def init_keyword_extractor(config):
    nlp_config = config.get('nlp')
    encoder_config = config.get('encoder')
    extractor_config = config.get('extractor')
    embedding_config = config.get('embedding')
    rank_config = config.get('rank')

    nlp = init_nlp(nlp_config)

    encoder = getattr(encoders, encoder_config.get('class'))(
        **encoder_config.get('kwargs')
    )

    extractor = getattr(extractors, extractor_config.get('class'))(
        **{"nlp": nlp, **extractor_config.get('kwargs')}
    )
    embedding = getattr(embeddings, embedding_config.get('class'))(
        **{"encoder": encoder, **embedding_config.get('kwargs')}
    )
    rank = getattr(ranks, rank_config.get('class'))(
        **rank_config.get('kwargs')
    )
    keyword_extractor = getattr(keyword_extractors, config.get('class'))(
        phrase_extractor=extractor, embed=embedding, rank=rank
    )

    return keyword_extractor
