import torch
from sentence_transformers import SentenceTransformer

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Device: {_device}')
_model_names = [
    'sentence-transformers/all-mpnet-base-v2'
]
_datasets = [
    #('msmarco', 'MSMARCO'),
    #('nq', 'NQ'),
    ('dbpedia-entity', 'DBPedia'),
    #('hotpotqa', 'HotpotQA'),
    #('trec-news', 'TREC-NEWS'),
    #('webis-touche2020', 'Touche 2020'),
    #('quora', 'Quora'),
    #('fever', 'FEVER'),
    #('scifact', 'SciFact'),
]


if __name__ == '__main__':
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    table_out = 'Model | ' + ' | '.join([ds_name for _, ds_name in _datasets]) + '\n'
    table_out += '----- | ' + ' | '.join(['-' * len(ds_name) for _, ds_name in _datasets]) + '\n'
    for model_name in _model_names:
        model = SentenceBERT(model_name, device=str(_device),
                             multi_emb=True)
        dres = DRES(model, batch_size=32)
        retriever = EvaluateRetrieval(dres, score_function="cos_sim_multiple")
        scores = []
        for dataset, ds_name in _datasets:
            print(f'Evaluating {ds_name}...')
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
            out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
            data_path = util.download_and_unzip(url, out_dir)

            #### Provide the data_path where scifact has been downloaded and unzipped
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

            #### Load the SBERT model and retrieve using cosine-similarity
            # model = SentenceBERT('sentence-transformers/all-distilroberta-v1', device=str(_device))
            results = retriever.retrieve(corpus, queries)

            #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            ndcg_10 = ndcg['NDCG@10']
            scores.append(ndcg_10)
        table_out += model_name + ' | ' + ' | '.join([f'{score * 100.0:.02f}' for score in scores]) + '\n'
    print('Result table follows:\n\n')
    print(table_out)
