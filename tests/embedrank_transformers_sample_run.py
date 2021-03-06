import spacy
from bert_serving.client import BertClient

from model.embedrank_transformers import EmbedRankTransformers

if __name__ == '__main__':
    bc = BertClient(output_fmt='list')
    nlp = spacy.load("en_core_web_lg", disable=['ner'])

    fi = EmbedRankTransformers(nlp=nlp,
                               dnn=bc,
                               perturbation='replacement',
                               emb_method='subtraction',
                               mmr_beta=0.55,
                               top_n=10,
                               alias_threshold=0.8)

    text = """
    Evaluation of existing and new feature recognition algorithms. 2. Experimental
	results
For pt.1 see ibid., p.839-851. This is the second of two papers investigating
	the performance of general-purpose feature detection techniques. The
	first paper describes the development of a methodology to synthesize
	possible general feature detection face sets. Six algorithms resulting
	from the synthesis have been designed and implemented on a SUN
	Workstation in C++ using ACIS as the geometric modelling system. In
	this paper, extensive tests and comparative analysis are conducted on
	the feature detection algorithms, using carefully selected components
	from the public domain, mostly from the National Design Repository. The
	results show that the new and enhanced algorithms identify face sets
	that previously published algorithms cannot detect. The tests also show
	that each algorithm can detect, among other types, a certain type of
	feature that is unique to it. Hence, most of the algorithms discussed
	in this paper would have to be combined to obtain complete coverage
    """

    marked_target, keywords, keyword_relevance = fi.fit(text)
    print(marked_target)
    print(f'Keywords: {keywords}')
    print(f'Keyword Relevance: {keyword_relevance}')

    print(fi.extract_keywords(text))
