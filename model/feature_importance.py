from typing import Tuple, Iterator, Dict, List

import spacy
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity


class FeatureImportance:
    """Implementation of Black-Box Feature importance. Finds important features in target text which causes high
    high similarity on query text.
         Feature: spaCy's document noun_chunks which are actually chunks of nouns that represent important parts of
         sentence.
         Black-Box: any model which gives good text embeddings optimized for cosine similarity search.
     """

    def __init__(self, nlp: spacy.language, bert: BertClient, loss: str = 'l1'):
        """Takes spaCy's language model, bert encoder model and loss parameter"""
        self.nlp = nlp
        self.bert = bert
        self.loss = loss
        self.pos_color = "0,255,0"
        self.neg_color = "255,0,0"
        self.min_alpha = 0.001

    def fit(self, query: str, target: str) -> Tuple[str, dict]:
        """Calculates feature importance on target text, then colors important features in text"""
        feature_importance = self.feature_importance(query, target)
        marked_target = self.mark_important_features(target, feature_importance)

        return marked_target, feature_importance

    def features(self, text: str) -> Iterator[Tuple[str, int, int]]:
        """Extracts features (noun chunks) from text and returns them with original indices"""
        for chunk in self.nlp(text).noun_chunks:
            yield (chunk.text, chunk.start_char, chunk.end_char)

    @staticmethod
    def perturb_feature(text: str, feature: Tuple[str, int, int]) -> str:
        """Removes given feature from text. (can be replaced with synonyms from different domain)"""
        return text[:feature[1]] + text[feature[2]:]

    def texts_and_features_perturbed(self, text: str) -> Iterator[Tuple[Tuple[str, int, int], str]]:
        """Iterates on target text and perturbs existing features in it (remove or replace)"""
        for feature in self.features(text):
            yield (feature, self.perturb_feature(text, feature))

    def feature_importance(self, query: str, target: str) -> Dict[Tuple[str, int, int], float]:
        """Calculates feature importance values in target text using cosine similarity loss normalization"""
        features_and_texts = [
            ((None, 0, 0), query),  # original query text to calculate it's embedding
            (('', 0, 0), target)  # original target text to calculate it's embedding
        ]

        # collect all extracted features (noun chunks) from text and it's perturbed version (feature removed)
        for feature, replaced_target in self.texts_and_features_perturbed(target):
            features_and_texts.append((feature, replaced_target))

        # calculate embeddings of all texts (including query and target texts) using BertClient in one batch
        embeddings = self.bert.encode([text for _, text, in features_and_texts])

        # get query and target embeddings
        query_emb = embeddings[0]
        target_emb = embeddings[1:]  # this includes original target text also

        # calculate cosine similarities between query embedding and target embeddings (original and perturbed ones)
        cosine_similarities = list(cosine_similarity([query_emb], target_emb)[0])

        # calculate normalized cosine losses (feature importance)
        feature_importance = {}
        for (feature, _), norm_loss in zip(features_and_texts[2:],
                                           self.normalize_similarity_losses(cosine_similarities)):
            feature_importance[feature] = norm_loss

        return feature_importance

    def cosine_loss(self, original: str, perturbed: str) -> float:
        """Calculates cosine loss between original and perturbed cosine similarity"""
        if self.loss == 'l1':
            return self.l1_loss_signed(original, perturbed)

        elif self.loss == 'l2':
            return self.l2_loss_signed(original, perturbed)

        else:
            raise ValueError(f"{self.loss} type of loss isn't supported!")

    @staticmethod
    def l1_loss_signed(original: str, perturbed: str) -> float:
        """L1 loss but saves sign (for negative and positive importance)"""
        return original - perturbed

    @staticmethod
    def l2_loss_signed(original: str, perturbed: str) -> float:
        """L2 loss but saves sign (for negative and positive importance)"""
        return (-1) * (perturbed > original) * (original - perturbed) ** 2

    def normalize_similarity_losses(self, cosine_similarities: List[float]) -> List[float]:
        """Normalized feature losses will give us some idea about feature importance. It can be negative or positive
        in range [-1,1]. -1 means that removal of that feature will increase cosine similarity between query and
        target document a lot. 1 means that removing of that feature will decrease the cosine similarity between them,
        hence it's very important for document matching and can be considered as good keyword.

        Args:
            cosine_similarities: cosine similarity between query and target (original and perturbed ones)

        Returns:
            normalized losses (feature importance)
        """
        # get cosine similarity between original query and target
        orig_cosine_similarity = cosine_similarities[0]

        # collect cosine similarities between query and perturbed targets
        cosine_losses = []
        for cos_sim in cosine_similarities[1:]:
            cosine_losses.append(self.cosine_loss(orig_cosine_similarity, cos_sim))

        # get minimum and maximum values of losses (can be negative and positive
        max_loss, min_loss = max(cosine_losses), min(cosine_losses)

        # loss will have the same sign but normalized in negative and positive ranges separately
        normalized_losses = []
        for loss in cosine_losses:
            if loss > 0:
                normalized_losses.append(loss / max_loss)
            elif loss < 0:
                normalized_losses.append((-1) * (loss / min_loss))
            else:
                normalized_losses.append(0.0)

        return normalized_losses

    def mark_color(self, feature: str, alpha: float) -> str:
        """Marks feature's background with it's relative color (red for negative and green for positive) with
        alpha (importance value)"""
        if self.min_alpha < alpha or alpha < - self.min_alpha:
            return f"<b style=\"background-color:rgba(" \
                f"{self.pos_color if alpha > 0 else self.neg_color},{alpha})\">" \
                f"{feature}</b>"
        else:
            return feature

    def mark_important_features(self, text: str, alphas: Dict[Tuple[str, int, int], float]) -> str:
        """marks features in given text using estimated alpha values (feature importance)"""
        marked_text = ''
        last_end = 0
        for (feature, st_idx, end_idx), alpha in alphas.items():
            marked_text += text[last_end:st_idx] + self.mark_color(feature, alpha)
            last_end = end_idx

        marked_text += text[last_end:]

        return marked_text
