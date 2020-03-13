# unsupervised_keyword_extraction
Using BERT pre-trained model embeddings for [EmbedRank](https://github.com/swisscom/ai-research-keyphrase-extraction) for unsupervised keyword extraction.

## Getting Started

### 1. Create environment
create conda environment with python 3.7 version
```bash
conda create --name keyword_extraction python=3.7
```

Activate environment
```bash
conda activate keyword_extraction
```

Install requirements
```bash
sh install_dependencies.sh
```

### 2. Download a pre-trained BERT model

<details>
 <summary>List of released pretrained BERT models (click to expand...)</summary>


<table>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip">BERT-Base, Uncased</a></td><td>12-layer, 768-hidden, 12-heads, 110M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip">BERT-Large, Uncased</a></td><td>24-layer, 1024-hidden, 16-heads, 340M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip">BERT-Base, Cased</a></td><td>12-layer, 768-hidden, 12-heads , 110M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip">BERT-Large, Cased</a></td><td>24-layer, 1024-hidden, 16-heads, 340M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip">BERT-Base, Multilingual Cased (New)</a></td><td>104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip">BERT-Base, Multilingual Cased (Old)</a></td><td>102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip">BERT-Base, Chinese</a></td><td>Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters</td></tr>
</table>

</details>

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip
```

### 3. Start Bert as a Service
```bash
sh run_bert_service.sh
```


### 4. Usage Example
```python
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
```

### 5. Evaluation
You can evaluate model on many different datasets using script bellow. See [here](run_evaluation.py) for mode details. (WARNING: if run_evaluation fails line 149, in build_printable
    printable[qrel] = pd.DataFrame(raw, columns=['app', *(table.columns.levels[1].get_values())[:-1]]) please replace `.get_values()` method with `.values` or downgrade pandas to some version that has it)
```bash
python -m run_evaluation
```

### 6. TrecEval Scores (Inspec)
```bash
Running Evaluation for Inspec dataset
                                app        F1_all           P_5       map_all
0                   Inspec_RAKE.out  0.147000 bl   0.274400 bl   0.116900 bl 
1                   Inspec_YAKE.out   0.136600 ▼    0.249200 ▼    0.103100 ▼ 
2  Inspec_EmbedRankTransformers.out   0.128300 ▼    0.238400 ▼    0.093400 ▼ 

```




## References
https://github.com/hanxiao/bert-as-service

https://www.groundai.com/project/embedrank-unsupervised-keyphrase-extraction-using-sentence-embeddings/1

https://monkeylearn.com/keyword-extraction/

https://arxiv.org/pdf/1801.04470.pdf

https://github.com/liaad/keep

https://github.com/LIAAD/KeywordExtractor-Datasets

https://spacy.io/usage/linguistic-features

https://github.com/usnistgov/trec_eval