import nltk
from nltk.corpus import stopwords


class PhraseExtractor:
    """Extracts candidate phrases from given text"""

    def __init__(self, nlp, text, np_rule='GRAMMAR1', method='regex'):
        """Takes nlp model (which supports POS tagging, SentTokenizer) and takes text to tokenize"""
        self.text = text
        self.nlp = nlp
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
        self.stopwords = set(stopwords.words('english'))
        self.method = method
        self.phrases = []
        self.color = '0,255,0'

        self.GRAMMAR1 = """  NP:{<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
        self.GRAMMAR2 = """  NP:{<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""
        self.GRAMMAR3 = """  NP:{<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

        self._init_np_parser(np_rule)
        self._process_text()
        self._extract_candidates()
        self._sort_candidates()

    def _init_np_parser(self, rule):
        self.np_parser = nltk.RegexpParser(self.__getattribute__(rule))

    def _process_text(self):
        self.doc = self.nlp(self.text)

        if self.method == 'regex':
            self.tagged_tokens = []

            for token in self.doc:
                if token.text.lower() in self.stopwords:
                    token.tag_ = 'IN'
                self.tagged_tokens.append((token.text.lower(), token.tag_, token.idx, token.idx + len(token)))

    def _extract_candidates(self):
        if self.method == 'regex':
            self.phrases = self._extract_candidates_regex()
        elif self.method == 'spacy':
            self.phrases = self._extract_candidates_spacy()

    def _extract_candidates_spacy(self):
        phrase_candidates = []

        for chunk in self.doc.noun_chunks:
            phrase_candidates.append((chunk.text.lower(), chunk.start_char, chunk.end_char))

        return phrase_candidates

    def _extract_candidates_regex(self):
        phrase_candidates = []
        np_tree = self.np_parser.parse(self.tagged_tokens)

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

        return phrase_candidates

    def _sort_candidates(self):
        self.phrases = sorted(self.phrases, key=lambda x: x[2])

    def mark_phrases(self):
        marked_text = ''
        last_end = 0
        for phrase, st_idx, end_idx in self.phrases:
            marked_text += self.text[last_end:st_idx] + self.mark_background(phrase, 1.0)
            last_end = end_idx

        marked_text += self.text[last_end:]

        return marked_text

    def mark_background(self, feature: str, alpha: float) -> str:
        return f"<b style=\"background-color:rgba({self.color},{alpha})\">{feature}</b>"
