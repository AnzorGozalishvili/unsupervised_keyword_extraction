pip install git+https://github.com/liaad/keep
pip install git+https://github.com/boudinfl/pke
pip install git+https://github.com/LIAAD/yake.git
python -m nltk.downloader stopwords
python -m spacy download en
python -m spacy download en_core_web_lg

# build trec_eval
mkdir temp_
cd temp_
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
# replace BIN variable which is path to binary where trec_eval should be installed
bin_path=$(which pip | sed 's+/pip++g')
sed -i "s+BIN = /usr/local/bin+BIN = $n+g" Makefile
make install
cd ../../
rm -rf temp_

# install requirements file
pip install -r requirements.txt

# download models
wget http://www.ccc.ipt.pt/~ricardo/keep/standalone/data.zip
unzip data.zip
rm data.zip