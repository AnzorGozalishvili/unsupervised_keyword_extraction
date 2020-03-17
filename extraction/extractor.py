import nltk
from nltk.corpus import stopwords


class NPGrammars:
    GRAMMAR1 = """  NP:{<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
    GRAMMAR2 = """  NP:{<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""
    GRAMMAR3 = """  NP:{<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


class NPMethods:
    NOUN_CHUNKS = "noun_chunks"
    GRAMMAR = "grammar"
    REGEX = 'regex'


class NPTags:
    NLTK = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}


class StopWords:
    NLTK = set(stopwords.words('english'))


class PhraseHighlighter:
    """Highlights phrases in text"""
    color = '0,255,0'

    @staticmethod
    def to_html(text, phrases):
        marked_text = ''
        last_end = 0
        for phrase, st_idx, end_idx in phrases:
            marked_text += text[last_end:st_idx] + PhraseHighlighter._highlight(phrase, 1.0)
            last_end = end_idx

        marked_text += text[last_end:]

        return marked_text

    @staticmethod
    def _highlight(phrase: str, alpha: float) -> str:
        return f"<b style=\"background-color:rgba({PhraseHighlighter.color},{alpha})\">{phrase}</b>"


class Extractor:
    """Extracts some slices from text and highlights them"""

    def __init__(self):
        pass

    def run(self, text):
        pass


class PhraseExtractor(Extractor):
    """Extracts candidate phrases from given text using language models"""

    def __init__(self, nlp, grammar='GRAMMAR1', np_method='NOUN_CHUNKS', np_tags='NLTK', stopwords="NLTK"):
        """Takes nlp model (which supports POS tagging, SentTokenizer) and takes text to tokenize"""
        super().__init__()

        self.method = getattr(NPMethods, np_method)
        self.considered_tags = getattr(NPTags, np_tags)
        self.stopwords = getattr(StopWords, stopwords)
        self.grammar = getattr(NPGrammars, grammar)
        self.nlp = nlp

        self._init_np_parser()

    def run(self, text):
        doc = self.nlp(text)

        if self.method == NPMethods.NOUN_CHUNKS:
            phrases = self._extract_candidates_spacy(doc)

        elif self.method == NPMethods.GRAMMAR:
            doc = self._override_stopword_tags(doc)
            tokens = self._extract_tokens(doc)
            phrases = self._extract_candidates_grammar(tokens)

        else:
            phrases = []

        return phrases

    def _init_np_parser(self):
        if self.method == NPMethods.GRAMMAR:
            self.np_parser = nltk.RegexpParser(self.grammar)

    def _override_stopword_tags(self, doc):
        if self.stopwords:
            for token in doc:
                if token.text.lower() in self.stopwords:
                    token.tag_ = 'IN'

        return doc

    @staticmethod
    def _extract_tokens(doc):
        return [(token.text.lower(), token.tag_, token.idx, token.idx + len(token)) for token in doc]

    @staticmethod
    def _extract_candidates_spacy(doc):
        phrase_candidates = []

        for chunk in doc.noun_chunks:
            phrase_candidates.append((chunk.text.lower(), chunk.start_char, chunk.end_char))

        return phrase_candidates

    def _extract_candidates_grammar(self, tokens):
        phrase_candidates = []
        np_tree = self.np_parser.parse(tokens)

        for node in np_tree:
            if isinstance(node, nltk.tree.Tree) and node._label == 'NP':
                tokens = []
                indices = set()
                for node_child in node.leaves():
                    tokens.append(node_child[0])
                    indices.add(node_child[2])
                    indices.add(node_child[3])

                phrase = ' '.join(tokens)

                phrase_start_idx = min(indices)
                phrase_end_idx = max(indices)

                phrase_candidates.append((phrase, phrase_start_idx, phrase_end_idx))

        sorted_phrase_candidates = self._sort_candidates(phrase_candidates)

        return sorted_phrase_candidates

    @staticmethod
    def _sort_candidates(phrases):
        return sorted(phrases, key=lambda x: x[2])
