import spacy
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage

from extraction.extractor import PhraseExtractor, PhraseHighlighter

if __name__ == '__main__':
    # Examples from SIFRank
    text_1 = "NuVox shows staying power with new cash, new market Who says you can't raise cash in today's telecom market? NuVox Communications positions itself for the long run with $78.5 million in funding and a new credit facility"
    text_2 = "This paper deals with two questions: Does social capital determine innovation in manufacturing firms? If it is the case, to what extent? To deal with these questions, we review the literature on innovation in order to see how social capital came to be added to the other forms of capital as an explanatory variable of innovation. In doing so, we have been led to follow the dominating view of the literature on social capital and innovation which claims that social capital cannot be captured through a single indicator, but that it actually takes many different forms that must be accounted for. Therefore, to the traditional explanatory variables of innovation, we have added five forms of structural social capital (business network assets, information network assets, research network assets, participation assets, and relational assets) and one form of cognitive social capital (reciprocal trust). In a context where empirical investigations regarding the relations between social capital and innovation are still scanty, this paper makes contributions to the advancement of knowledge in providing new evidence regarding the impact and the extent of social capital on innovation at the two decisionmaking stages considered in this study"

    # stanfordnlp.download('en')
    nlp = spacy.load('en_core_web_sm')
    corenlp = StanfordNLPLanguage(stanfordnlp.Pipeline(lang="en"))

    spacy_native = PhraseExtractor(nlp, np_method='NOUN_CHUNKS')
    spacy_grammar = PhraseExtractor(nlp, grammar='GRAMMAR1', np_method='NOUN_CHUNKS', np_tags='NLTK', stopwords='NLTK')
    corenlp_grammar = PhraseExtractor(corenlp, grammar='GRAMMAR1', np_method='NOUN_CHUNKS', np_tags='NLTK',
                                      stopwords='NLTK')

    # SHOW RESULTS
    # grammar method (corenlp tags)
    print(PhraseHighlighter.to_html(text_1, corenlp_grammar.run(text_1)))
    print(PhraseHighlighter.to_html(text_2, corenlp_grammar.run(text_2)))

    # grammar method (spacy tags)
    print(PhraseHighlighter.to_html(text_1, spacy_grammar.run(text_1)))
    print(PhraseHighlighter.to_html(text_2, spacy_grammar.run(text_2)))

    # spacy method (spacy tags)
    print(PhraseHighlighter.to_html(text_1, spacy_native.run(text_1)))
    print(PhraseHighlighter.to_html(text_2, spacy_native.run(text_2)))
