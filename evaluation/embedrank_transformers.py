import os

from keep.utility import getlanguage, CreateKeywordsFolder, LoadFiles, Convert2TrecEval

from helpers import read_json
from keyword_extraction.helpers import init_keyword_extractor


class EmbedRankTransformers(object):
    def __init__(self, numOfKeywords, pathData, dataset_name, normalization, model=None):
        self.__lan = getlanguage(pathData + "/Datasets/" + dataset_name)
        self.__numOfKeywords = numOfKeywords
        self.__dataset_name = dataset_name
        self.__normalization = normalization
        self.__pathData = pathData
        self.__pathToDFFile = self.__pathData + "/Models/Unsupervised/dfs/" + self.__dataset_name + '_dfs.gz'
        self.__pathToDatasetName = self.__pathData + "/Datasets/" + self.__dataset_name
        self.__keywordsPath = self.__pathData + f'/Keywords/{self.__class__.__name__}/' + self.__dataset_name
        self.__outputPath = self.__pathData + "/conversor/output/"
        self.__algorithmName = self.__class__.__name__

        self.model = model

    def LoadDatasetFiles(self):
        # Gets all files within the dataset fold
        listFile = LoadFiles(self.__pathToDatasetName + '/docsutf8/*')
        print(f"\ndatasetID = {self.__dataset_name}; Number of Files = "
              f"{len(listFile)}; Language of the Dataset = {self.__lan}")
        return listFile

    def CreateKeywordsOutputFolder(self):
        # Set the folder where keywords are going to be saved
        CreateKeywordsFolder(self.__keywordsPath)

    def runSingleDoc(self, doc):
        try:
            # read raw document
            with open(doc, 'r') as doc_reader:
                doc_text = doc_reader.read()

            # extract keywords
            keywords, relevance = self.model.run(doc_text)
            keywords = [(keyword, score) for (keyword, _, _), score in zip(keywords, relevance)]
        except:
            keywords = []

        return keywords

    def runMultipleDocs(self, listOfDocs):
        self.CreateKeywordsOutputFolder()

        for j, doc in enumerate(listOfDocs):
            # docID keeps the name of the file (without the extension)
            docID = '.'.join(os.path.basename(doc).split('.')[0:-1])

            keywords = self.runSingleDoc(doc)

            # Save the keywords; score (on Algorithms/NameOfAlg/Keywords/NameOfDataset
            with open(os.path.join(self.__keywordsPath, docID), 'w', encoding="utf-8") as out:
                for (key, score) in keywords:
                    out.write(f'{key} {score}\n')

            # Track the status of the task
            print(f"\rFile: {j + 1}/{len(listOfDocs)}", end='')

        print(f"\n100% of the Extraction Concluded")

    def ExtractKeyphrases(self):
        # print(f"\n------------------------------Compute DF--------------------------")
        # self.ComputeDocumentFrequency()

        print(f"\n\n-----------------Extract Keyphrases--------------------------")
        listOfDocs = self.LoadDatasetFiles()
        self.runMultipleDocs(listOfDocs)

    def Convert2Trec_Eval(self, EvaluationStemming=False):
        Convert2TrecEval(self.__pathToDatasetName, EvaluationStemming, self.__outputPath, self.__keywordsPath,
                         self.__dataset_name, self.__algorithmName)


class EmbedRankBERT(EmbedRankTransformers):
    def __init__(self, numOfKeywords, pathData, dataset_name, normalization):
        super().__init__(numOfKeywords, pathData, dataset_name, normalization)
        self.__keywordsPath = self.__pathData + f'/Keywords/{self.__class__.__name__}/' + self.__dataset_name
        self.__algorithmName = self.__class__.__name__
        self.model = init_keyword_extractor(read_json('evaluation/config/embedrank_bert_as_a_service.json'))


class EmbedRankSentenceBERT(EmbedRankTransformers):
    def __init__(self, numOfKeywords, pathData, dataset_name, normalization):
        super().__init__(numOfKeywords, pathData, dataset_name, normalization)
        self.__keywordsPath = self.__pathData + f'/Keywords/{self.__class__.__name__}/' + self.__dataset_name
        self.__algorithmName = self.__class__.__name__
        self.model = init_keyword_extractor(read_json('evaluation/config/embedrank_sentence_bert.json'))
