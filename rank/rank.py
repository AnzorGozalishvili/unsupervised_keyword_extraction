import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Rank:
    def __init__(self):
        pass

    def run(self, text, phrases, text_embedding, phrase_embeddings):
        pass


class EmbedMethods:
    NAIVE = 'naive'
    EXTRACTIVE = 'extractive'
    SUBTRACTIVE = 'subtractive'


class EmbedRank:
    """Implementation of unsupervised `phrase` extraction method using DNN embeddings and MMR. This method tries to
    Find important phrases in text using analysis of their cosine similarity to original text and using Maximum
    Marginal Relevance method to choose most relevant and also diverse phrases.

         phrase: i.e. `noun phrases` from (nltk, spacy, corenlp, etc) which are actually chunks of nouns that represent
         important parts of sentence. This is assumed to be good selection of candidates for phrases.
         DNN: any model which gives good text embeddings optimized for cosine similarity search."""

    def __init__(self, emb_method='NAIVE', mmr_beta=0.55, top_n=10, alias_threshold=0.8):
        """Takes spaCy's language model, dnn encoder model and loss parameter"""
        self.emb_method = getattr(EmbedMethods, emb_method)
        self.min_alpha = 0.001
        self.mmr_beta = mmr_beta
        self.top_n = top_n
        self.alias_threshold = alias_threshold

    def run(self, text, phrases, text_emb, phrase_embs):
        top_phrases, relevance, aliases = self.mmr(text_emb, phrases, phrase_embs, self.mmr_beta,
                                                   self.top_n, self.alias_threshold)

        return top_phrases, relevance, aliases

    def mmr(self, text_emb, phrases, phrase_embs, beta=0.55, top_n=10, alias_threshold=0.8):
        """Implementation of Maximal Marginal Relevance to get top N relevant phrases to text

        Args:
            text_emb: embedding of original text (from where phrases are extracted)
            phrases: phrases (noun phrases) selected from text from where we have to choose broad and relevant phrases
            phrase_embs: embeddings of given phrases
            beta: hyperparameter for MMR score calculations (controls tradeoff between informativeness and diversity)
            top_n: number of top phrases to extract (will return less phrases if len(phrases) < top_n)
            alias_threshold: threshold for cosine similarities (controls alias phrase pairs assignment)

        Returns:
            top_phrases: selected top phrases
            relevance: relevance values for these phrases (relevance of phrase to original text)
            aliases_phrases: aliases for each phrase
        """
        # calculate similarities of phrases with text and between phrases
        text_sims = cosine_similarity(phrase_embs, [text_emb])
        phrase_sims = cosine_similarity(phrase_embs)

        # normalize cosine similarities
        text_sims_norm = self.standard_normalize_cosine_similarities(text_sims)
        phrase_sims_norm = self.max_normalize_cosine_similarities_pairwise(phrase_sims)

        # keep indices of selected and unselected phrases in list
        selected_phrase_indices = []
        unselected_phrase_indices = list(range(len(phrases)))

        # find the most similar phrase (using original cosine similarities)
        best_idx = np.argmax(text_sims)
        selected_phrase_indices.append(best_idx)
        unselected_phrase_indices.remove(best_idx)

        # do top_n - 1 cycle to select top N phrases
        for _ in range(min(len(phrases), top_n) - 1):
            unselected_phrase_distances_to_text = text_sims_norm[unselected_phrase_indices, :]
            unselected_phrase_distances_pairwise = phrase_sims_norm[unselected_phrase_indices][:,
                                                   selected_phrase_indices]

            # if dimension of phrases distances is 1 we add additional axis to the end
            if unselected_phrase_distances_pairwise.ndim == 1:
                unselected_phrase_distances_pairwise = np.expand_dims(unselected_phrase_distances_pairwise, axis=1)

            # find new candidate with
            idx = int(np.argmax(
                beta * unselected_phrase_distances_to_text - (1 - beta) * np.max(unselected_phrase_distances_pairwise,
                                                                                 axis=1).reshape(-1, 1)))
            best_idx = unselected_phrase_indices[idx]

            # select new best phrase and update selected/unselected phrase indices list
            selected_phrase_indices.append(best_idx)
            unselected_phrase_indices.remove(best_idx)

        # calculate relevance using original (not normalized) cosine similarities of phrases to text
        relevance = self.max_normalize_cosine_similarities(text_sims[selected_phrase_indices]).tolist()
        aliases_phrases = self.get_alias_phrases(phrase_sims[selected_phrase_indices, :], phrases, alias_threshold)

        top_phrases = [phrases[idx] for idx in selected_phrase_indices]

        return top_phrases, relevance, aliases_phrases

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
    def get_alias_phrases(phrase_sims, phrases, threshold):
        """Find phrases in selected list that are aliases (very similar) to each other"""
        similarities = np.nan_to_num(phrase_sims, 0)
        sorted_similarities = np.flip(np.argsort(similarities), 1)

        aliases = []
        for idx, item in enumerate(sorted_similarities):
            alias_for_item = []
            for i in item:
                if similarities[idx, i] >= threshold:
                    alias_for_item.append(phrases[i])
                else:
                    break
            aliases.append(alias_for_item)

        return aliases
