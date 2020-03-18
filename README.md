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

### 6. TrecEval Score Results
----------------------------------------------------------------------------------------
Running Evaluation for Inspec
```markdown
|FIELD1|app                             |F1_10       |F1_15       |F1_5        |P_10        |P_15        |P_5         |map_10      |map_15      |map_5       |recall_10   |recall_15   |recall_5    |
|------|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|0     |Inspec_RAKE.out                 |0.206600 bl |0.220100 bl |0.152400 bl |0.250400 bl |0.216900 bl |0.282300 bl |0.100100 bl |0.115100 bl |0.070500 bl |0.188100 bl |0.236900 bl |0.110300 bl |
|1     |Inspec_YAKE.out                 |0.176300 ▼  |0.187800 ▼  |0.144500 ▼  |0.208300 ▼  |0.181400 ▼  |0.261700 ▼  |0.092000 ▼  |0.104000 ▼  |0.072700    |0.165800 ▼  |0.214100 ▼  |0.105400 ᐁ  |
|2     |Inspec_MultiPartiteRank.out     |0.186600 ▼  |0.201300 ▼  |0.156000    |0.221000 ▼  |0.190600 ▼  |0.285600    |0.101700    |0.114100    |0.081100 ▲  |0.171200 ▼  |0.216700 ▼  |0.113000    |
|3     |Inspec_TopicalPageRank.out      |0.226800 ▲  |0.241000 ▲  |0.174100 ▲  |0.272700 ▲  |0.233700 ▲  |0.319600 ▲  |0.116500 ▲  |0.133500 ▲  |0.084200 ▲  |0.206600 ▲  |0.257900 ▲  |0.126100 ▲  |
|4     |Inspec_TopicRank.out            |0.178000 ▼  |0.186800 ▼  |0.149000    |0.211100 ▼  |0.175300 ▼  |0.272300    |0.093900 ▼  |0.103000 ▼  |0.075100 ᐃ  |0.161300 ▼  |0.195600 ▼  |0.107800    |
|5     |Inspec_SingleRank.out           |0.224200 ▲  |0.237900 ▲  |0.170900 ▲  |0.269600 ▲  |0.231400 ▲  |0.313500 ▲  |0.114400 ▲  |0.131200 ▲  |0.082600 ▲  |0.204800 ▲  |0.256300 ▲  |0.123800 ▲  |
|6     |Inspec_TextRank.out             |0.123500 ▼  |0.127200 ▼  |0.097500 ▼  |0.140900 ▼  |0.106500 ▼  |0.177800 ▼  |0.050600 ▼  |0.052900 ▼  |0.040900 ▼  |0.102100 ▼  |0.113100 ▼  |0.068900 ▼  |
|7     |Inspec_KPMiner.out              |0.013400 ▼  |0.013400 ▼  |0.013300 ▼  |0.011700 ▼  |0.007800 ▼  |0.022900 ▼  |0.006600 ▼  |0.006600 ▼  |0.006600 ▼  |0.008400 ▼  |0.008400 ▼  |0.008200 ▼  |
|8     |Inspec_TFIDF.out                |0.135900 ▼  |0.153800 ▼  |0.100400 ▼  |0.157100 ▼  |0.146000 ▼  |0.176400 ▼  |0.059300 ▼  |0.069900 ▼  |0.043900 ▼  |0.129700 ▼  |0.178100 ▼  |0.074100 ▼  |
|9     |Inspec_KEA.out                  |0.123000 ▼  |0.134900 ▼  |0.095200 ▼  |0.142700 ▼  |0.128700 ▼  |0.166600 ▼  |0.053600 ▼  |0.061300 ▼  |0.041300 ▼  |0.117400 ▼  |0.156100 ▼  |0.070500 ▼  |
|10    |Inspec_EmbedRankTransformers.out|0.231000 ▲  |0.231000 ▲  |0.175300 ▲  |0.278600 ▲  |0.185700 ▼  |0.328400 ▲  |0.117500 ▲  |0.117500    |0.084900 ▲  |0.206500 ▲  |0.206500 ▼  |0.125900 ▲  |
```
----------------------------------------------------------------------------------------
Running Evaluation for SemEval2017
```markdown
|    | app                                   | F1_10       | F1_15       | F1_5        | P_10        | P_15        | P_5         | map_10      | map_15      | map_5       | recall_10   | recall_15   | recall_5    |
|----|---------------------------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 0  | SemEval2017_RAKE.out                  | 0.216700 bl | 0.246500 bl | 0.140200 bl | 0.299600 bl | 0.272200 bl | 0.309500 bl | 0.093700 bl | 0.114600 bl | 0.058200 bl | 0.179000 bl | 0.240200 bl | 0.093700 bl |
| 1  | SemEval2017_YAKE.out                  | 0.172000 ▼  | 0.199500 ▼  | 0.114000 ▼  | 0.235900 ▼  | 0.219300 ▼  | 0.249100 ▼  | 0.073400 ▼  | 0.088400 ▼  | 0.049900 ▼  | 0.143400 ▼  | 0.196300 ▼  | 0.076600 ▼  |
| 2  | SemEval2017_MultiPartiteRank.out      | 0.213100    | 0.238600    | 0.161600 ▲  | 0.297000    | 0.264200    | 0.358600 ▲  | 0.106400 ▲  | 0.125700 ᐃ  | 0.077000 ▲  | 0.175600    | 0.231900    | 0.108100 ▲  |
| 3  | SemEval2017_TopicalPageRank.out       | 0.253100 ▲  | 0.289400 ▲  | 0.173000 ▲  | 0.350900 ▲  | 0.319300 ▲  | 0.382200 ▲  | 0.124600 ▲  | 0.152900 ▲  | 0.081500 ▲  | 0.208700 ▲  | 0.281400 ▲  | 0.115900 ▲  |
| 4  | SemEval2017_TopicRank.out             | 0.203300 ᐁ  | 0.222400 ▼  | 0.159600 ▲  | 0.285600    | 0.247600 ▼  | 0.357800 ▲  | 0.100500    | 0.116500    | 0.075300 ▲  | 0.166300 ᐁ  | 0.213400 ▼  | 0.106300 ▲  |
| 5  | SemEval2017_SingleRank.out            | 0.248100 ▲  | 0.286300 ▲  | 0.170000 ▲  | 0.343800 ▲  | 0.316400 ▲  | 0.373200 ▲  | 0.120700 ▲  | 0.149300 ▲  | 0.078600 ▲  | 0.204500 ▲  | 0.278000 ▲  | 0.114100 ▲  |
| 6  | SemEval2017_TextRank.out              | 0.132800 ▼  | 0.149200 ▼  | 0.091300 ▼  | 0.185000 ▼  | 0.158200 ▼  | 0.206500 ▼  | 0.050100 ▼  | 0.057100 ▼  | 0.035400 ▼  | 0.107000 ▼  | 0.134700 ▼  | 0.060700 ▼  |
| 7  | SemEval2017_KPMiner.out               | 0.032200 ▼  | 0.032200 ▼  | 0.032000 ▼  | 0.034100 ▼  | 0.022900 ▼  | 0.066900 ▼  | 0.016300 ▼  | 0.016400 ▼  | 0.016100 ▼  | 0.018900 ▼  | 0.019100 ▼  | 0.018700 ▼  |
| 8  | SemEval2017_TFIDF.out                 | 0.166900 ▼  | 0.180200 ▼  | 0.131500    | 0.235500 ▼  | 0.200900 ▼  | 0.297400    | 0.076700 ▼  | 0.087500 ▼  | 0.058100    | 0.137200 ▼  | 0.175500 ▼  | 0.087600    |
| 9  | SemEval2017_KEA.out                   | 0.151800 ▼  | 0.160200 ▼  | 0.122200 ▼  | 0.214000 ▼  | 0.178400 ▼  | 0.276300 ᐁ  | 0.069400 ▼  | 0.077400 ▼  | 0.053800    | 0.124700 ▼  | 0.156200 ▼  | 0.081600 ▼  |
| 10 | SemEval2017_EmbedRankTransformers.out | 0.249400 ▲  | 0.249400    | 0.165800 ▲  | 0.345400 ▲  | 0.230300 ▼  | 0.370400 ▲  | 0.117400 ▲  | 0.117400    | 0.076000 ▲  | 0.204200 ▲  | 0.204200 ▼  | 0.110300 ▲  |
```

### 7. SIFRank Evaluation Scores 
F1 Scores on `N=5` (first N extracted keywords)

| Models       | Inspec       | SemEval2017   | DUC2001      |
| :-----       | :----:       | :----:        |:----:        |
| TFIDF        | 11.28        | 12.70         |  9.21        |
| YAKE         | 15.73        | 11.84         | 10.61        |
| TextRank     | 24.39        | 16.43         | 13.94        |
| SingleRank   | 24.69        | 18.23         | 21.56        |
| TopicRank    | 22.76        | 17.10         | 20.37        |
| PositionRank | 25.19        | 18.23         | 24.95        |
| Multipartite | 23.05        | 17.39         | 21.86        |
| RVA          | 21.91        | 19.59         | 20.32        |
| **Model_v1** |`23.31`       | `14.60`       | N/A          |
| EmbedRank d2v| 27.20        | 20.21         | 21.74        |
| SIFRank      | **29.11**    | **22.59**     | 24.27        |
| SIFRank+     | 28.49        | 21.53         | **30.88**    |


## References
https://github.com/hanxiao/bert-as-service

https://www.groundai.com/project/embedrank-unsupervised-keyphrase-extraction-using-sentence-embeddings/1

https://monkeylearn.com/keyword-extraction/

https://arxiv.org/pdf/1801.04470.pdf

https://github.com/liaad/keep

https://github.com/LIAAD/KeywordExtractor-Datasets

https://spacy.io/usage/linguistic-features

https://github.com/usnistgov/trec_eval