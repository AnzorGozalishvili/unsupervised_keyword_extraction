import re
from typing import Tuple

import numpy as np
import unidecode


class PerturbMethods:
    REMOVE = 'remove'
    REPLACE = 'replace'


class Pooling:
    MEAN = 'mean'
    MAX = 'max'
    MIN = 'min'


class Embedding:
    def __init__(self, encoder):
        self.encoder = encoder

    def run(self, text, phrases):
        pass


class NaiveEmbedding(Embedding):
    def __init__(self, encoder):
        super().__init__(encoder)

    def run(self, text, phrases):
        embeddings = self.encoder.encode([text for text, _, _ in [(text, 0, -1)] + phrases])

        text_emb = np.array(embeddings[0])
        phrase_embs = np.array(embeddings[1:])

        return text_emb, phrase_embs


class ExtractiveEmbedding(Embedding):
    def __init__(self, encoder, pooling='MEAN'):
        super().__init__(encoder)
        self.pooling = getattr(Pooling, pooling)

    def run(self, text, phrases):
        text_emb, text_tokens = self.encoder.encode([text], show_tokens=True)
        text_emb = text_emb[0]
        text_tokens = text_tokens[0]

        phrase_embs = []
        for phrase, _, _ in phrases:
            phrase_embs.append(self.get_phrase_embedding(phrase.lower(), text_emb, text_tokens))

        text_emb = self.pool(text_emb[:len(text_tokens), :])
        phrase_embs = phrase_embs

        return text_emb, phrase_embs

    @staticmethod
    def filter_token(token):
        return re.sub(r"[#| ]", "", token)

    def pool(self, arr):
        if self.pooling == Pooling.MEAN:
            return self.pool_mean(arr)
        elif self.pooling == Pooling.MAX:
            return self.pool_max(arr)
        elif self.pooling == Pooling.MIN:
            return self.pool_min(arr)

    @staticmethod
    def pool_mean(arr):
        return arr.mean(axis=0)

    @staticmethod
    def pool_max(arr):
        return arr.max(axis=0)

    @staticmethod
    def pool_min(arr):
        return arr.min(axis=0)

    def merge_tokens(self, sent_tokens):
        text = ''
        indices = []
        for idx, token in enumerate(sent_tokens):
            if token not in ['[CLS]', '[SEP]']:
                filtered_token = self.filter_token(token)
                text += filtered_token
                indices += [idx] * len(filtered_token)

        return text, indices

    @staticmethod
    def get_all_unique_matching_indices(query, text, indices):
        unique_matching_indices = set()
        for match in re.finditer(re.escape(query), text):
            unique_matching_indices.update(indices[match.start():match.end()])

        if not unique_matching_indices:
            print(f'NO MATCH!  {query}:{text}')

        return unique_matching_indices

    def get_token_indices(self, phrase, sent_tokens):
        sent_text, sent_indices = self.merge_tokens(sent_tokens)
        phrase_text = unidecode.unidecode_expect_nonascii(self.filter_token(phrase))
        match_indices = self.get_all_unique_matching_indices(phrase_text, sent_text, sent_indices)

        return list(match_indices)

    def get_phrase_embedding(self, phrase, text_embs, text_tokens):
        token_indices = self.get_token_indices(phrase, text_tokens)
        phrase_emb = self.pool(text_embs[token_indices, :])

        return phrase_emb


class SubtractiveEmbedding(Embedding):
    def __init__(self, encoder, method='REMOVE'):
        super().__init__(encoder)
        self.method = getattr(PerturbMethods, method)

    def run(self, text, phrases):
        # create list that holds phrases, it's start and end indices and text
        texts = [text]

        for phrase in phrases:
            texts.append(self.perturb_keyword(text, phrase))

        # calculate embeddings of all texts using DDN encoder in one batch
        embeddings = self.encoder.encode(texts)

        # get original (non-perturbed) and perturbed text embeddings
        text_emb = np.array(embeddings[0])
        perturbed_embs = np.array(embeddings[1:])

        # calculate keyword embedding with subtraction method (subtract original text with keyword perturbed text)
        # it seems to be getting good keyword vectors and is experimental approach for keyword embeddings
        phrase_embs = text_emb - perturbed_embs

        return text_emb, phrase_embs

    def perturb_keyword(self, text: str, phrase: Tuple[str, int, int]) -> str:
        """Perturbs given phrase in text. (replaces with general noun or removes)"""
        if self.method == PerturbMethods.REMOVE:
            return text[:phrase[1]] + text[phrase[2]:]
        elif self.method == PerturbMethods.REPLACE:
            return text[:phrase[1]] + 'that' + text[phrase[2]:]
        else:
            raise ValueError(f'Unsupported perturbation method {self.method}')
