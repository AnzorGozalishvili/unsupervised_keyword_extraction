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
```html
----------------------------------------------------------------------------------------
Running Evaluation for [1mInspec[0m dataset
<table border="1" class="dataframe">
    <thead>
    <tr style="text-align: right;">
        <th></th>
        <th>app</th>
        <th>F1_10</th>
        <th>F1_15</th>
        <th>F1_5</th>
        <th>P_10</th>
        <th>P_15</th>
        <th>P_5</th>
        <th>map_10</th>
        <th>map_15</th>
        <th>map_5</th>
        <th>recall_10</th>
        <th>recall_15</th>
        <th>recall_5</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <th>0</th>
        <td>Inspec_RAKE.out</td>
        <td>0.206600 bl</td>
        <td>0.220100 bl</td>
        <td>0.152400 bl</td>
        <td>0.250400 bl</td>
        <td>0.216900 bl</td>
        <td>0.282300 bl</td>
        <td>0.100100 bl</td>
        <td>0.115100 bl</td>
        <td>0.070500 bl</td>
        <td>0.188100 bl</td>
        <td>0.236900 bl</td>
        <td>0.110300 bl</td>
    </tr>
    <tr>
        <th>1</th>
        <td>Inspec_YAKE.out</td>
        <td>0.176300 ▼</td>
        <td>0.187700 ▼</td>
        <td>0.144500 ▼</td>
        <td>0.208300 ▼</td>
        <td>0.181400 ▼</td>
        <td>0.261700 ▼</td>
        <td>0.092000 ▼</td>
        <td>0.104000 ▼</td>
        <td>0.072700</td>
        <td>0.165800 ▼</td>
        <td>0.214100 ▼</td>
        <td>0.105400 ᐁ</td>
    </tr>
    <tr>
        <th>2</th>
        <td>Inspec_MultiPartiteRank.out</td>
        <td>0.186600 ▼</td>
        <td>0.201300 ▼</td>
        <td>0.156000</td>
        <td>0.221000 ▼</td>
        <td>0.190600 ▼</td>
        <td>0.285600</td>
        <td>0.101700</td>
        <td>0.114100</td>
        <td>0.081100 ▲</td>
        <td>0.171200 ▼</td>
        <td>0.216600 ▼</td>
        <td>0.113000</td>
    </tr>
    <tr>
        <th>3</th>
        <td>Inspec_TopicalPageRank.out</td>
        <td>0.226800 ▲</td>
        <td>0.241000 ▲</td>
        <td>0.174100 ▲</td>
        <td>0.272700 ▲</td>
        <td>0.233700 ▲</td>
        <td>0.319600 ▲</td>
        <td>0.116500 ▲</td>
        <td>0.133500 ▲</td>
        <td>0.084200 ▲</td>
        <td>0.206600 ▲</td>
        <td>0.257900 ▲</td>
        <td>0.126100 ▲</td>
    </tr>
    <tr>
        <th>4</th>
        <td>Inspec_TopicRank.out</td>
        <td>0.177900 ▼</td>
        <td>0.186800 ▼</td>
        <td>0.149000</td>
        <td>0.211100 ▼</td>
        <td>0.175300 ▼</td>
        <td>0.272300</td>
        <td>0.093800 ▼</td>
        <td>0.103000 ▼</td>
        <td>0.075100 ᐃ</td>
        <td>0.161300 ▼</td>
        <td>0.195600 ▼</td>
        <td>0.107800</td>
    </tr>
    <tr>
        <th>5</th>
        <td>Inspec_SingleRank.out</td>
        <td>0.224200 ▲</td>
        <td>0.237900 ▲</td>
        <td>0.170900 ▲</td>
        <td>0.269600 ▲</td>
        <td>0.231400 ▲</td>
        <td>0.313500 ▲</td>
        <td>0.114400 ▲</td>
        <td>0.131200 ▲</td>
        <td>0.082600 ▲</td>
        <td>0.204800 ▲</td>
        <td>0.256300 ▲</td>
        <td>0.123800 ▲</td>
    </tr>
    <tr>
        <th>6</th>
        <td>Inspec_TextRank.out</td>
        <td>0.123500 ▼</td>
        <td>0.127200 ▼</td>
        <td>0.097500 ▼</td>
        <td>0.140900 ▼</td>
        <td>0.106500 ▼</td>
        <td>0.177800 ▼</td>
        <td>0.050600 ▼</td>
        <td>0.052900 ▼</td>
        <td>0.040900 ▼</td>
        <td>0.102100 ▼</td>
        <td>0.113100 ▼</td>
        <td>0.068900 ▼</td>
    </tr>
    <tr>
        <th>7</th>
        <td>Inspec_EmbedRankTransformers.out</td>
        <td>0.231000 ▲</td>
        <td>0.231000 ▲</td>
        <td>0.175300 ▲</td>
        <td>0.278600 ▲</td>
        <td>0.185700 ▼</td>
        <td>0.328400 ▲</td>
        <td>0.117400 ▲</td>
        <td>0.117400</td>
        <td>0.084900 ▲</td>
        <td>0.206500 ▲</td>
        <td>0.206500 ▼</td>
        <td>0.125900 ▲</td>
    </tr>
    </tbody>
</table>


----------------------------------------------------------------------------------------
Running Evaluation for [1mSemEval2017[0m dataset
<table border="1" class="dataframe">
    <thead>
    <tr style="text-align: right;">
        <th></th>
        <th>app</th>
        <th>F1_10</th>
        <th>F1_15</th>
        <th>F1_5</th>
        <th>P_10</th>
        <th>P_15</th>
        <th>P_5</th>
        <th>map_10</th>
        <th>map_15</th>
        <th>map_5</th>
        <th>recall_10</th>
        <th>recall_15</th>
        <th>recall_5</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <th>0</th>
        <td>SemEval2017_RAKE.out</td>
        <td>0.216700 bl</td>
        <td>0.246500 bl</td>
        <td>0.140200 bl</td>
        <td>0.299600 bl</td>
        <td>0.272200 bl</td>
        <td>0.309500 bl</td>
        <td>0.093700 bl</td>
        <td>0.114600 bl</td>
        <td>0.058200 bl</td>
        <td>0.179000 bl</td>
        <td>0.240200 bl</td>
        <td>0.093700 bl</td>
    </tr>
    <tr>
        <th>1</th>
        <td>SemEval2017_YAKE.out</td>
        <td>0.171900 ▼</td>
        <td>0.199500 ▼</td>
        <td>0.114000 ▼</td>
        <td>0.235900 ▼</td>
        <td>0.219300 ▼</td>
        <td>0.249100 ▼</td>
        <td>0.073400 ▼</td>
        <td>0.088400 ▼</td>
        <td>0.049900 ▼</td>
        <td>0.143300 ▼</td>
        <td>0.196300 ▼</td>
        <td>0.076600 ▼</td>
    </tr>
    <tr>
        <th>2</th>
        <td>SemEval2017_MultiPartiteRank.out</td>
        <td>0.213100</td>
        <td>0.238600</td>
        <td>0.161600 ▲</td>
        <td>0.297000</td>
        <td>0.264200</td>
        <td>0.358600 ▲</td>
        <td>0.106400 ▲</td>
        <td>0.125700 ᐃ</td>
        <td>0.077000 ▲</td>
        <td>0.175600</td>
        <td>0.231900</td>
        <td>0.108100 ▲</td>
    </tr>
    <tr>
        <th>3</th>
        <td>SemEval2017_TopicalPageRank.out</td>
        <td>0.253100 ▲</td>
        <td>0.289400 ▲</td>
        <td>0.173000 ▲</td>
        <td>0.350900 ▲</td>
        <td>0.319300 ▲</td>
        <td>0.382200 ▲</td>
        <td>0.124600 ▲</td>
        <td>0.152900 ▲</td>
        <td>0.081500 ▲</td>
        <td>0.208700 ▲</td>
        <td>0.281400 ▲</td>
        <td>0.115900 ▲</td>
    </tr>
    <tr>
        <th>4</th>
        <td>SemEval2017_TopicRank.out</td>
        <td>0.203300 ᐁ</td>
        <td>0.222400 ▼</td>
        <td>0.159600 ▲</td>
        <td>0.285600</td>
        <td>0.247600 ▼</td>
        <td>0.357800 ▲</td>
        <td>0.100500</td>
        <td>0.116500</td>
        <td>0.075300 ▲</td>
        <td>0.166300 ᐁ</td>
        <td>0.213400 ▼</td>
        <td>0.106200 ▲</td>
    </tr>
    <tr>
        <th>5</th>
        <td>SemEval2017_SingleRank.out</td>
        <td>0.248100 ▲</td>
        <td>0.286300 ▲</td>
        <td>0.170000 ▲</td>
        <td>0.343800 ▲</td>
        <td>0.316400 ▲</td>
        <td>0.373200 ▲</td>
        <td>0.120700 ▲</td>
        <td>0.149300 ▲</td>
        <td>0.078600 ▲</td>
        <td>0.204500 ▲</td>
        <td>0.278000 ▲</td>
        <td>0.114000 ▲</td>
    </tr>
    <tr>
        <th>6</th>
        <td>SemEval2017_TextRank.out</td>
        <td>0.132800 ▼</td>
        <td>0.149300 ▼</td>
        <td>0.091300 ▼</td>
        <td>0.185000 ▼</td>
        <td>0.158400 ▼</td>
        <td>0.206500 ▼</td>
        <td>0.050100 ▼</td>
        <td>0.057100 ▼</td>
        <td>0.035400 ▼</td>
        <td>0.107000 ▼</td>
        <td>0.134700 ▼</td>
        <td>0.060700 ▼</td>
    </tr>
    <tr>
        <th>7</th>
        <td>SemEval2017_EmbedRankTransformers.out</td>
        <td>0.249400 ▲</td>
        <td>0.249400</td>
        <td>0.165700 ▲</td>
        <td>0.345400 ▲</td>
        <td>0.230300 ▼</td>
        <td>0.370400 ▲</td>
        <td>0.117400 ▲</td>
        <td>0.117400</td>
        <td>0.076000 ▲</td>
        <td>0.204200 ▲</td>
        <td>0.204200 ▼</td>
        <td>0.110300 ▲</td>
    </tr>
    </tbody>
</table>

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