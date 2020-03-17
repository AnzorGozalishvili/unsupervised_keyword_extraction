class KeywordExtractor:
    """Selects candidate phrases from input text, calculates their embeddings and applies Ranking algorithm
    to extract relevant keywords from text"""

    def __init__(self, phrase_extractor, embed, rank):
        self.phrase_extractor = phrase_extractor
        self.embed = embed
        self.rank = rank

    def run(self, text):
        phrases = self.phrase_extractor.run(text)
        text_embedding, phrase_embeddings = self.embed.run(text, phrases)
        ranked_phrases, phrase_relevance, phrase_aliases = self.rank.run(text, phrases, text_embedding,
                                                                         phrase_embeddings)

        return ranked_phrases, phrase_relevance
