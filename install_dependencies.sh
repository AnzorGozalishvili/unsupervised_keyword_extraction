pip install git+https://github.com/liaad/keep
pip install git+https://github.com/boudinfl/pke
pip install git+https://github.com/LIAAD/yake.git
python -m nltk.downloader stopwords
python -m spacy download en
python -m spacy download en_core_web_lg

# build trec_eval
git clone git@github.com:usnistgov/trec_eval.git
cd trec_eval
make install
cd ..
rm -rf trec_eval

pip install -r requirements.txt
