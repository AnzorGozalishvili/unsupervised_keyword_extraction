import os

from IPython.display import display
from keep import KEA
from keep import KPMiner
from keep import MultiPartiteRank
from keep import PositionRank
from keep import Rake
from keep import SIGTREC_Eval
from keep import SingleRank
from keep import TFIDF
from keep import TextRank
from keep import TopicRank
from keep import TopicalPageRank
from keep import YAKE

from evaluation.embedrank_transformers import EmbedRankBERT
from evaluation.embedrank_transformers import EmbedRankSentenceBERT
from evaluation.embedrank import EmbedRank as ER
from evaluation.sifrank import SIFRank as SR
from evaluation.sifrankplus import SIFRankPlus as SRP


def keyword_extraction():
    for algorithm in ListOfAlgorithms:
        print("\n")
        print("----------------------------------------------------------------------------------------")
        print(f"Preparing Evaluation for \033[1m{algorithm}\033[0m algorithm")

        for i in range(len(ListOfDatasets)):
            dataset_name = ListOfDatasets[i]
            print("\n----------------------------------")
            print(f" dataset_name = {dataset_name}")
            print("----------------------------------")

            if algorithm == 'RAKE':
                Rake_object = Rake(numOfKeyphrases, pathData, dataset_name)
                Rake_object.ExtractKeyphrases()
                Rake_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'YAKE':
                YAKE_object = YAKE(numOfKeyphrases, pathData, dataset_name)
                YAKE_object.ExtractKeyphrases()
                YAKE_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'MultiPartiteRank':
                MultiPartiteRank_object = MultiPartiteRank(numOfKeyphrases, pathData, dataset_name)
                MultiPartiteRank_object.ExtractKeyphrases()
                MultiPartiteRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TopicalPageRank':
                TopicalPageRank_object = TopicalPageRank(numOfKeyphrases, pathData, dataset_name, normalization)
                TopicalPageRank_object.ExtractKeyphrases()
                TopicalPageRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TopicRank':
                TopicRank_object = TopicRank(numOfKeyphrases, pathData, dataset_name)
                TopicRank_object.ExtractKeyphrases()
                TopicRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'PositionRank':
                PositionRank_object = PositionRank(numOfKeyphrases, pathData, dataset_name, normalization)
                PositionRank_object.ExtractKeyphrases()
                PositionRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'SingleRank':
                SingleRank_object = SingleRank(numOfKeyphrases, pathData, dataset_name, normalization)
                SingleRank_object.ExtractKeyphrases()
                SingleRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TextRank':
                TextRank_object = TextRank(numOfKeyphrases, pathData, dataset_name, normalization)
                TextRank_object.ExtractKeyphrases()
                TextRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'KPMiner':
                KPMiner_object = KPMiner(numOfKeyphrases, pathData, dataset_name, normalization)
                KPMiner_object.ExtractKeyphrases()
                KPMiner_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TFIDF':
                TFIDF_object = TFIDF(numOfKeyphrases, pathData, dataset_name, normalization)
                TFIDF_object.ExtractKeyphrases()
                TFIDF_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'KEA':
                KEA_object = KEA(numOfKeyphrases, pathData, dataset_name, normalization)
                KEA_object.ExtractKeyphrases(nFolds)
                KEA_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'EmbedRankBERT':
                EmbedRankBERT_object = EmbedRankBERT(numOfKeyphrases, pathData, dataset_name, normalization)
                EmbedRankBERT_object.ExtractKeyphrases()
                EmbedRankBERT_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'EmbedRankSentenceBERT':
                EmbedRankSentenceBERT_object = EmbedRankSentenceBERT(numOfKeyphrases, pathData, dataset_name,
                                                                     normalization)
                EmbedRankSentenceBERT_object.ExtractKeyphrases()
                EmbedRankSentenceBERT_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'EmbedRank':
                ER_object = ER(numOfKeyphrases, pathData, dataset_name, normalization)
                ER_object.ExtractKeyphrases()
                ER_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'SIFRank':
                SR_object = SR(numOfKeyphrases, pathData, dataset_name, normalization)
                SR_object.ExtractKeyphrases()
                SR_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'SIFRankPlus':
                SRP_object = SRP(numOfKeyphrases, pathData, dataset_name, normalization)
                SRP_object.ExtractKeyphrases()
                SRP_object.Convert2Trec_Eval(EvaluationStemming)


def evaluation():
    for dataset in ListOfDatasets:
        print("\n")
        print("----------------------------------------------------------------------------------------")
        print(f"Running Evaluation for \033[1m{dataset}\033[0m dataset")

        path2qrel_file = f"{pathOutput}{dataset}.qrel"
        datasetid = os.path.basename(path2qrel_file)

        resultsFiles = []
        for alg in ListOfAlgorithms:
            resultsFiles.append(f"{pathOutput}{dataset}_{alg}.out")

        sig = SIGTREC_Eval()
        results = sig.Evaluate(path2qrel_file, datasetid, resultsFiles, measures, statistical_test, formatOutput)

        for res in results:
            if formatOutput == "df":
                display(res)
            else:
                print(res)


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Some algorithms have a normalization parameter which may be defined with None, stemming or lemmatization
    normalization = None  # Other options: "stemming" (porter) and "lemmatization"

    # Num of Keyphrases do Retrieve
    numOfKeyphrases = 20

    # Num of folds for training KEA
    nFolds = 5

    # ListOfDatasets = ['110-PT-BN-KP', '500N-KPCrowd-v1.1', 'citeulike180',
    #                   'fao30', 'fao780', 'Inspec', 'kdd', 'Krapivin2009',
    #                   'Nguyen2007', 'pak2018', 'PubMed', 'Schutz2008', 'SemEval2010',
    #                   'SemEval2017', 'theses100', 'wiki20', 'www', 'cacic', 'wicc', 'WikiNews']
    ListOfDatasets = ['Inspec', 'SemEval2017']

    # ListOfAlgorithms = ['RAKE', 'YAKE', 'MultiPartiteRank', 'TopicalPageRank', 'TopicRank', 'SingleRank', 'TextRank',
    #                    'KPMiner', 'TFIDF', 'KEA', 'EmbedRank', 'SIFRank', 'SIFRankPlus', 'EmbedRankBERT', 'EmbedRankSentenceBERT']

    ListOfAlgorithms = ['EmbedRankSentenceBERT']

    pathData = 'data'
    pathOutput = pathData + "/conversor/output/"
    pathDataset = pathData + "/Datasets/"
    pathKeyphrases = pathData + "/Keywords/"

    EvaluationStemming = True  # performs stemming when comparing the results obtained from the algorithm with the ground-truth

    statistical_test = ["student"]  # wilcoxon

    measures = [
        'F1.5', 'P.5', 'recall.5', 'map.5',
        'F1.10', 'P.10', 'recall.10', 'map.10',
        'F1.15', 'P.15', 'recall.15', 'map.15',
        'F1', 'map'
    ]

    formatOutput = 'csv'  # options: 'csv', 'html', 'json', 'latex', 'sql', 'string', 'df'
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # run keyword extraction
    # ------------------------------------------------------------------------------------------------------------------
    keyword_extraction()

    # ------------------------------------------------------------------------------------------------------------------
    # run evaluation
    # ------------------------------------------------------------------------------------------------------------------
    evaluation()
