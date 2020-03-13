from typing import Tuple, Iterator, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sns.set()


class EmbedRankTransformers:
    """Implementation of unsupervised keyword extraction method using DNN embeddings and MMR. This method tries to
    Find important keywords in text using analysis of their cosine similarity to original text and using Maximum
    Marginal Relevance method to choose most relevant and also diverse keywords.

         keyword: spaCy's document noun_chunks (noun phrases) which are actually chunks of nouns that represent
         important parts of sentence. This is assumed to be good selection of candidates for keywords.
         DNN: any model which gives good text embeddings optimized for cosine similarity search."""

    def __init__(self, nlp: spacy.language, dnn: Any, perturbation='removal', emb_method='subtraction', mmr_beta=0.55,
                 top_n=10, alias_threshold=0.8):
        """Takes spaCy's language model, dnn encoder model and loss parameter"""
        self.nlp = nlp
        self.dnn = dnn
        self.perturbation = perturbation
        self.emb_method = emb_method
        self.pos_color = "0,255,0"
        self.neg_color = "255,0,0"
        self.min_alpha = 0.001
        self.mmr_beta = mmr_beta
        self.top_n = top_n
        self.alias_threshold = alias_threshold

    def fit(self, text: str) -> Tuple[str, list, list]:
        """Finds top keywords and then colors them according to their relevance to text"""
        keywords, keyword_relevance, aliases = self.keyword_relevance(text)
        marked_target = self.mark_keywords_relevance(text, keywords, keyword_relevance)

        return marked_target, keywords, keyword_relevance

    def keywords(self, text: str) -> Iterator[Tuple[str, int, int]]:
        """Extracts keywords (noun phrases) from text and returns them with original indices"""
        for chunk in self.nlp(text).noun_chunks:
            yield (chunk.text, chunk.start_char, chunk.end_char)

    def perturb_keyword(self, text: str, keyword: Tuple[str, int, int]) -> str:
        """Removes given keyword from text. (can be replaced with synonyms from different domain)"""
        if self.perturbation == 'removal':
            return text[:keyword[1]] + text[keyword[2]:]
        elif self.perturbation == 'replacement':
            return text[:keyword[1]] + 'that' + text[keyword[2]:]
        else:
            raise ValueError(f'Unsupported perturbation method {self.perturbation}')

    def texts_and_keywords_perturbed(self, text: str) -> Iterator[Tuple[Tuple[str, int, int], str]]:
        """Iterates on text and perturbs existing keywords in it (remove or replace)"""
        for keyword in self.keywords(text):
            yield (keyword, self.perturb_keyword(text, keyword))

    def keyword_relevance(self, text: str) -> Tuple[List[Tuple[str, int, int]], List[float], List[str]]:
        """Calculates keyword relevance values in text"""
        # extract keywords from text, calculates text and keywords embeddings
        text_embedding, keyword_embeddings, perturbed_embs, keywords = self.get_text_and_keyword_embeddings(text)

        # apply MMR to find top keywords
        top_keywords, relevance, aliases = self.mmr(text_embedding, keywords, keyword_embeddings, self.mmr_beta,
                                                    self.top_n, self.alias_threshold)

        # get indices of keywords according to their occurrence in text
        sorted_indices = self.get_ordered_keywords_indices(top_keywords)

        # reorder all results according to keyword sorted indices
        top_keywords = self.get_elements(top_keywords, sorted_indices)
        relevance = self.get_elements(relevance, sorted_indices)
        aliases = self.get_elements(aliases, sorted_indices)

        return top_keywords, relevance, aliases

    @staticmethod
    def get_elements(elements, indices):
        return [elements[idx] for idx in indices]

    @staticmethod
    def get_ordered_keywords_indices(keywords):
        """Gives indices of sorted keywords (according to their start indices"""
        return np.argsort(np.array(keywords)[:, 1].astype(int)).tolist()

    def get_text_and_keyword_embeddings(self, text):
        """Calculates text and keyword embeddings."""
        if self.emb_method == 'subtraction':

            # create list that holds keywords, it's start and end indices and text
            keywords_and_texts = [
                ((None, 0, 0), text),  # original text to calculate it's embedding
            ]

            # collect all extracted keywords (noun phrases) from text and it's perturbed text version (keyword removed)
            for keyword, perturbed_keyword_text in self.texts_and_keywords_perturbed(text):
                keywords_and_texts.append((keyword, perturbed_keyword_text))

            # calculate embeddings of all texts using DDN encoder in one batch
            embeddings = self.dnn.encode([text for _, text, in keywords_and_texts])

            # get original (non-perturbed) and perturbed text embeddings
            text_emb = np.array(embeddings[0])
            perturbed_embs = np.array(embeddings[1:])

            # calculate keyword embedding with subtraction method (subtract original text with keyword perturbed text)
            # it seems to be getting good keyword vectors and is experimental approach for keyword embeddings
            keyword_embs = text_emb - perturbed_embs

            # get list of keywords
            keywords = [keyword for keyword, _, in keywords_and_texts[1:]]

        elif self.emb_method == 'naive':

            # create list that holds keywords, it's start and end indices and text
            keywords_and_texts = [
                ((None, 0, 0), text),  # original text to calculate it's embedding
            ]

            # collect all extracted keywords (noun phrases) from their texts
            for keyword in self.keywords(text):
                keywords_and_texts.append((keyword, keyword[0]))

            # calculate embeddings of all texts using DDN encoder in one batch
            embeddings = self.dnn.encode([text for _, text, in keywords_and_texts])

            # get text and keywords embeddings
            text_emb = np.array(embeddings[0])
            keyword_embs = np.array(embeddings[1:])

            # get list of keywords
            keywords = [keyword for keyword, _, in keywords_and_texts[1:]]

            perturbed_embs = text_emb - keyword_embs

        else:
            raise ValueError(f"{self.emb_method} embedding method isn't supported")

        return text_emb, keyword_embs, perturbed_embs, keywords

    def mark_color(self, feature: str, alpha: float) -> str:
        """Marks feature's background with it's relative color (red for negative and green for positive) with
        alpha (importance value)"""
        if self.min_alpha < alpha or alpha < - self.min_alpha:
            return f"<b style=\"background-color:rgba(" \
                f"{self.pos_color if alpha > 0 else self.neg_color},{alpha})\">" \
                f"{feature}</b>"
        else:
            return feature

    def mark_keywords_relevance(self, text: str, keywords: List[Tuple[str, int, int]], alphas: List[float]) -> str:
        """marks keywords in given text using estimated alpha values (relevance score)"""
        marked_text = ''
        last_end = 0
        for (keyword, st_idx, end_idx), alpha in zip(keywords, alphas):
            marked_text += text[last_end:st_idx] + self.mark_color(keyword, alpha)
            last_end = end_idx

        marked_text += text[last_end:]

        return marked_text

    def mmr(self, text_emb, keywords, keyword_embs, beta=0.55, top_n=10, alias_threshold=0.8):
        """Implementation of Maximal Marginal Relevance to get top N relevant keywords to text

        Args:
            text_emb: embedding of original text (from where keywords are extracted)
            keywords: keywords (noun phrases) selected from text from where we have to choose broad and relevant keywords
            keyword_embs: embeddings of given keywords
            beta: hyperparameter for MMR score calculations (controls tradeoff between informativeness and diversity)
            top_n: number of top keywords to extract (will return less keywords if len(keywords) < top_n)
            alias_threshold: threshold for cosine similarities (controls alias keyword pairs assignment)

        Returns:
            top_keywords: selected top keywords
            relevance: relevance values for these keywords (relevance of keyword to original text)
            aliases_keywords: aliases for each keyword
        """
        # calculate similarities of keywords with text and between keywords
        text_sims = cosine_similarity(keyword_embs, [text_emb])
        keyword_sims = cosine_similarity(keyword_embs)

        # normalize cosine similarities
        text_sims_norm = self.standard_normalize_cosine_similarities(text_sims)
        keyword_sims_norm = self.max_normalize_cosine_similarities_pairwise(keyword_sims)

        # keep indices of selected and unselected keywords in list
        selected_keyword_indices = []
        unselected_keyword_indices = list(range(len(keywords)))

        # find the most similar keyword (using original cosine similarities)
        best_idx = np.argmax(text_sims)
        selected_keyword_indices.append(best_idx)
        unselected_keyword_indices.remove(best_idx)

        # do top_n - 1 cycle to select top N keywords
        for _ in range(min(len(keywords), top_n) - 1):
            unselected_keyword_distances_to_text = text_sims_norm[unselected_keyword_indices, :]
            unselected_keyword_distances_pairwise = keyword_sims_norm[unselected_keyword_indices][:,
                                                    selected_keyword_indices]

            # if dimension of keywords distances is 1 we add additional axis to the end
            if unselected_keyword_distances_pairwise.ndim == 1:
                unselected_keyword_distances_pairwise = np.expand_dims(unselected_keyword_distances_pairwise, axis=1)

            # find new candidate with
            idx = int(np.argmax(
                beta * unselected_keyword_distances_to_text - (1 - beta) * np.max(unselected_keyword_distances_pairwise,
                                                                                  axis=1).reshape(-1, 1)))
            best_idx = unselected_keyword_indices[idx]

            # select new best keyword and update selected/unselected keyword indices list
            selected_keyword_indices.append(best_idx)
            unselected_keyword_indices.remove(best_idx)

        # calculate relevance using original (not normalized) cosine similarities of keywords to text
        relevance = self.max_normalize_cosine_similarities(text_sims[selected_keyword_indices]).tolist()
        aliases_keywords = self.get_alias_keywords(keyword_sims[selected_keyword_indices, :], keywords, alias_threshold)

        top_keywords = [keywords[idx] for idx in selected_keyword_indices]

        # for showing vectors in space
        embs = keyword_embs + [text_emb]
        texts = keywords + ["DOC"]
        mask = []

        self.plot()

        return top_keywords, relevance, aliases_keywords

    @staticmethod
    def standard_normalize_cosine_similarities(cosine_similarities):
        """Normalized cosine similarities"""
        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.min(cosine_similarities)) / (
                np.max(cosine_similarities) - np.min(cosine_similarities))

        # standardize and shift by 0.5
        cosine_sims_norm = 0.5 + (cosine_sims_norm - np.mean(cosine_sims_norm)) / np.std(cosine_sims_norm)

        return cosine_sims_norm

    @staticmethod
    def max_normalize_cosine_similarities_pairwise(cosine_similarities):
        """Normalized cosine similarities of pairs which is 2d matrix of pairwise cosine similarities"""
        cosine_sims_norm = np.copy(cosine_similarities)
        np.fill_diagonal(cosine_sims_norm, np.NaN)

        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.nanmin(cosine_similarities, axis=0)) / (
                np.nanmax(cosine_similarities, axis=0) - np.nanmin(cosine_similarities, axis=0))

        # standardize shift by 0.5
        cosine_sims_norm = \
            0.5 + (cosine_sims_norm - np.nanmean(cosine_sims_norm, axis=0)) / np.nanstd(cosine_sims_norm, axis=0)

        return cosine_sims_norm

    @staticmethod
    def max_normalize_cosine_similarities(cosine_similarities):
        """Normalize cosine similarities using max normalization approach"""
        return 1 / np.max(cosine_similarities) * cosine_similarities.squeeze(axis=1)

    @staticmethod
    def get_alias_keywords(keyword_sims, keywords, threshold):
        """Find keywords in selected list that are aliases (very similar) to each other"""
        similarities = np.nan_to_num(keyword_sims, 0)
        sorted_similarities = np.flip(np.argsort(similarities), 1)

        aliases = []
        for idx, item in enumerate(sorted_similarities):
            alias_for_item = []
            for i in item:
                if similarities[idx, i] >= threshold:
                    alias_for_item.append(keywords[i])
                else:
                    break
            aliases.append(alias_for_item)

        return aliases

    @staticmethod
    def plot(embs, texts, mask):
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(embs)

        data = pd.DataFrame(
            {'v1': vectors_2d[:, 0],
             'v2': vectors_2d[:, 1],
             'type': mask
             }
        )

        ax = sns.scatterplot(x=data.v1, y=data.v2, style=data.type, hue=data.type)

        for i, text in enumerate(zip(texts)):
            if len(text) > 20:
                text = text[:20] + '...'
            ax.annotate(text, (vectors_2d[i, 0], vectors_2d[i, 1]))

        plt.show()
