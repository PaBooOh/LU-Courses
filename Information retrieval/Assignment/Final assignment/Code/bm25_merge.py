from statistics import median
from textwrap import indent
import ir_datasets
import numpy as np
from ir_datasets.datasets.clinicaltrials import ClinicalTrialsDocs, ClinicalTrialsDoc
from elasticsearch import Elasticsearch
import pandas as pd

from time import time
import json
import pytrec_eval

es = Elasticsearch()
dataset: ClinicalTrialsDocs = ir_datasets.load(
    "clinicaltrials/2021/trec-ct-2021")
# (hyper) parameters
N = dataset.docs_count()
RETRIEVE_SIZE = 10000
k1 = 1.2
b = 0.75

INDEX_DATASET_NAME = "clinical-trial-track"
INDEX_QUERIES_NAME = "ctt-queries"
DATASET_MAPPING_SIMPLE = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        }
    },
    "mappings": {
        "properties": {
            "doc_id": {
                "type": "text"
            },
            "summary": {
                "type": "text",
                "fielddata": "true",
                "term_vector": "with_positions_offsets_payloads",
                "store": "true",
                "analyzer": "whitespace"
            }
        }
    }
}
DATASET_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        }
    },
    "mappings": {
        "properties": {
            "doc_id": {
                "type": "text"
            },
            "title": {
                "type": "text",
                "fielddata": "true",
                "term_vector": "with_positions_offsets_payloads",
                "store": "true",
                "analyzer": "whitespace"
            },
            "condition": {
                "type": "text",
                "fielddata": "true",
                "term_vector": "with_positions_offsets_payloads",
                "store": "true",
                "analyzer": "whitespace"
            },
            "summary": {
                "type": "text",
                "fielddata": "true",
                "term_vector": "with_positions_offsets_payloads",
                "store": "true",
                "analyzer": "whitespace"
            },
            "detaiiled_description": {
                "type": "text",
                "fielddata": "true",
                "term_vector": "with_positions_offsets_payloads",
                "store": "true",
                "analyzer": "whitespace"
            },
            "eligibility": {
                "type": "text",
                "fielddata": "true",
                "term_vector": "with_positions_offsets_payloads",
                "store": "true",
                "analyzer": "whitespace"
            }
        }
    }
}

def define_dataset_fields(doc_id, title=None, condition=None, summary=None, detailed_description=None, eligibility=None, simple=True):
    if simple:
        doc = {
            'doc_id': doc_id,
            'summary': summary
        }
    else:
        doc = {
            'doc_id': doc_id,
            'title': title,
            'condition': condition,
            'summary': summary,
            'detailed_description': detailed_description,
            'eligibility': eligibility
        }
    return doc

def merge_fields(doc):
        merge = doc.title + doc.condition + doc.summary + doc.detailed_description + doc.eligibility
        return merge
        # print('title', doc.title)
        # print('condition', doc.condition)
        # print('summary', doc.summary)
        # print('desc', doc.detailed_description)
        # print('elig', doc.eligibility)
        
def index_dataset_docs(simple=True):
    # namedtuple<doc_id, title, condition, summary, detailed_description, eligibility>
    for idx, doc in enumerate(dataset.docs_iter()):
        if not simple:
            doc = define_dataset_fields(
                doc.doc_id, doc.title, doc.condition, doc.summary, doc.detailed_description, doc.eligibility)
        else:
            merge = merge_fields(doc)
            doc = define_dataset_fields(
                doc.doc_id, summary=merge, simple=True)
        es.index(index=INDEX_DATASET_NAME, body=doc, id=idx)

def create_dataset_index_with_mapping():
    es.indices.create(index=INDEX_DATASET_NAME, body=DATASET_MAPPING_SIMPLE)
    index_dataset_docs()

def index_queries_docs():
    dataset = pd.read_csv('queries_2021.tsv', sep='\t', header=None)
    ids, queries = dataset[0], dataset[1]
    for id, query in zip(ids, queries):
        doc = {
            'query_id': id,
            'query': query
        }
        es.index(index=INDEX_QUERIES_NAME, body=doc, id=id)

def define_query_body(query_text):
    query_body = {
        "size": RETRIEVE_SIZE,
        "query": {
            "bool": {
                "should": [
                    {"match":
                     {"summary": query_text}
                     }],
                "minimum_should_match": 1,
                "boost": 1.0
            }
        }
    }
    return query_body

# search result by default
def search_by_query(query_id=1):
    query_result = es.get(index=INDEX_QUERIES_NAME, id=query_id)
    query_text = query_result['_source']['query']
    query_body = define_query_body(query_text)
    doc_list = es.search(index=INDEX_DATASET_NAME, body=query_body)
    return doc_list['hits']['hits']

def default_es_score():
    score_dic = {}
    start = time()
    for query_id in range(1, 76):
        doc_list = search_by_query(query_id=query_id)
        for doc in doc_list:
            score_dic.setdefault(str(query_id), {})[
                doc['_source']['doc_id']] = doc['_score']
        if query_id % 10 == 0:
            print(
                f'walking to query {query_id} spends {round((time()-start)/60,2)} minutes')
    return score_dic

def delete_index(index='clinical-trial-track'):
    es.indices.delete(index=index)

def get_indices_name():
    indices = es.indices.get(index='*')
    return indices.keys()

def get_splitted_query(query_id=1):
    '''seperate by space'''
    query_text = es.get(index=INDEX_QUERIES_NAME, id=query_id)[
        '_source']['query']
    return query_text.split(' ')

def get_field_statistics(index=INDEX_DATASET_NAME, doc_id=1, field='summary', terms=False):
    term_vectors = es.termvectors(index=index, id=doc_id, term_statistics=True)
    sum_doc_freq = term_vectors['term_vectors'][field]['field_statistics']["sum_doc_freq"]
    doc_count = term_vectors['term_vectors'][field]['field_statistics']["doc_count"]
    sum_ttf = term_vectors['term_vectors'][field]['field_statistics']["sum_ttf"]
    if terms:
        terms = term_vectors['term_vectors'][field]['terms']
        return sum_doc_freq, doc_count, sum_ttf, terms
    return sum_doc_freq, doc_count, sum_ttf

def get_Lavg():
    _, doc_count, sum_ttf = get_field_statistics(terms=False)
    # print(1.0 * sum_ttf / doc_count)
    return 1.0 * sum_ttf / doc_count

def get_Ld(doc_id, field='summary'):
    '''return: length of each doc summary content string number'''
    doc = es.get(index=INDEX_DATASET_NAME, id=doc_id)['_source']
    text: str = doc[field]
    splitted_text = text.split()
    return len(splitted_text)

def get_doc_ids_by_query(query_id=1):
    docs = search_by_query(query_id)
    doc_ids = [doc['_id'] for doc in docs]
    return doc_ids

def get_relevance_judgements(binary=True) -> dict:
    '''
    Format of relevance data is https://trec.nist.gov/data/qrels_eng/
    return: topic and its documents relevance
    '''
    standard = pd.read_csv('./qrels2021.txt', sep=' ', header=None,
                           names=['topic', 'iteration', 'document#', 'relevance'])

    if binary:
        standard = standard.copy()
        standard['relevance'] = standard['relevance'].replace(2, 1)

    query_dic = {}
    for ind, item in standard.iterrows():
        # notice: for only dictionary as input of evaluation
        query_dic.setdefault(str(item['topic']), {})[
            item['document#']] = item['relevance']
    return query_dic

def score_store_in_local(dic, name, indent=None):
    '''
    store score in local to save time for thirty minutes
    '''
    jsons = json.dumps(dic, indent=indent)
    with open(name, 'w') as file:
        file.write(jsons)

def get_score_from_json(file) -> dict:
    '''
    get data from local file
    '''
    with open(file) as f:
        dic_score = json.load(f)
    print(type(dic_score))
    return dic_score

class Lucene_accurate:
    def __init__(self) -> None:
        self.L_avg = get_Lavg()

    def compute_first_term(self, df):
        idf = np.log(1 + ((N - df + 0.5) / (df + 0.5)))
        return idf

    def compute_second_term(self, tf, L_d):
        L_avg = self.L_avg
        numerator = tf
        denominator = k1 * (1 - b + (b * (L_d / L_avg))) + tf
        return numerator / denominator

    def compute_single_score(self, df, tf, L_d):
        first_term = self.compute_first_term(df)
        second_term = self.compute_second_term(tf, L_d)
        return first_term * second_term

    def compute_aggregate_score_for_query_doc(self, splitted_query, doc_id, terms):
        L_d = get_Ld(doc_id)
        scores = 0
        for query_term in splitted_query:
            # _, _, _, terms = get_field_statistics(doc_id=doc_id, terms=True)
            statistics = terms.get(query_term)
            if statistics is None:
                score = 0
            else:
                df, tf = statistics['doc_freq'], statistics['term_freq']
                score = self.compute_single_score(df, tf, L_d)
            scores += score
        return scores

    # target at each query!!
    def compute_aggregate_score_for_query_docs(self, query_id, field='summary'):
        ''''
        return: docs_score: dictionary with doc# as key, score for all splitted query as score
        '''
        splitted_query = get_splitted_query(query_id=query_id)
        doc_ids = get_doc_ids_by_query(query_id=query_id)
        docs_score = {}
        for j in range(25):
            # mterm_vector
            mterm_vectors = es.mtermvectors(
                index=INDEX_DATASET_NAME, ids=doc_ids[400*j:400*(j+1)], term_statistics=True)["docs"]
            # 400 doc
            for i in range(400):
                doc_id = mterm_vectors[i]["_id"]
                # content
                terms = mterm_vectors[i]["term_vectors"][field]["terms"]
                scores = self.compute_aggregate_score_for_query_doc(
                    splitted_query, doc_id, terms)
                # find doc
                doc_num = es.get(index=INDEX_DATASET_NAME, id=doc_id)[
                    '_source']['doc_id']
                docs_score[doc_num] = scores
        return docs_score

    def compute_aggregate_score_for_queries_docs(self):
        ''''
        return: evaluate with key of query_id, value of score of each doc
        '''
        evaluate = {}
        start = time()
        for query_id in range(1, 76):
            docs_score = self.compute_aggregate_score_for_query_docs(
                query_id=query_id)
            evaluate[str(query_id)] = docs_score
            if query_id % 5 == 0:
                print(
                    f'walking to query {query_id} spends {round((time()-start)/60,2)} minutes')
        # print(json.dumps(evaluate,indent=4))
        return evaluate

class Atire:
    def __init__(self) -> None:
        self.L_avg = get_Lavg()

    def compute_first_term(self, df):
        idf = np.log(N / df)
        return idf

    def compute_second_term(self, tf, L_d):
        L_avg = self.L_avg
        numerator = (k1 + 1) * tf
        denominator = k1 * (1 - b + (b * (L_d / L_avg))) + tf
        return numerator / denominator

    def compute_single_score(self, df, tf, L_d):
        first_term = self.compute_first_term(df)
        second_term = self.compute_second_term(tf, L_d)
        return first_term * second_term

    def compute_aggregate_score_for_query_doc(self, splitted_query, doc_id, terms):
        L_d = get_Ld(doc_id)
        scores = 0
        for query_term in splitted_query:
            # _, _, _, terms = get_field_statistics(doc_id=doc_id, terms=True)
            statistics = terms.get(query_term)
            if statistics is None:
                score = 0
            else:
                df, tf = statistics['doc_freq'], statistics['term_freq']
                score = self.compute_single_score(df, tf, L_d)
            scores += score
        return scores

    def compute_aggregate_score_for_query_docs(self, query_id, field='summary'):
        splitted_query = get_splitted_query(query_id=query_id)
        doc_ids = get_doc_ids_by_query(query_id=query_id)
        docs_score = {}
        for j in range(25):
            mterm_vectors = es.mtermvectors(
                index=INDEX_DATASET_NAME, ids=doc_ids[400*j:400*(j+1)], term_statistics=True)["docs"]
            for i in range(400):
                doc_id = mterm_vectors[i]["_id"]
                terms = mterm_vectors[i]["term_vectors"][field]["terms"]
                scores = self.compute_aggregate_score_for_query_doc(
                    splitted_query, doc_id, terms)
                doc_num = es.get(index=INDEX_DATASET_NAME, id=doc_id)[
                    '_source']['doc_id']
                docs_score[doc_num] = scores
        return docs_score

    def compute_aggregate_score_for_queries_docs(self):
        evaluate = {}
        start = time()
        for query_id in range(1, 5):
            docs_score = self.compute_aggregate_score_for_query_docs(
                query_id=query_id)
            evaluate[str(query_id)] = docs_score
            if query_id % 10 == 0:
                print(
                    f'walking to query {query_id} spends {round((time()-start)/60,2)} minutes')
        # print(json.dumps(evaluate,indent=4))
        # print(len(evaluate))
        return evaluate

def evaluate(run):
    from statistics import median
    # read standard judgement
    judge = get_relevance_judgements(binary=False)
    judge_binary = get_relevance_judgements(binary=True)

    # get results to be evaluated form local file
    #run = get_score_from_json(file=file)

    # for official metrics
    evaluator_off = pytrec_eval.RelevanceEvaluator(
        judge, {'ndcg_cut.5', 'ndcg_cut.10'})

    res_off = evaluator_off.evaluate(run)
    #print(json.dumps(res_off, indent=4))

    ndcg_cut_5 = median([value['ndcg_cut_5']
                        for value in res_off.values()])
    ndcg_cut_10 = median([value['ndcg_cut_10']
                         for value in res_off.values()])
    print(f'ndcg_cut_5 is {ndcg_cut_5}')
    print(f'ndcg_cut_10 is {ndcg_cut_10}')

    # for binary metrics
    evaluator_bina = pytrec_eval.RelevanceEvaluator(
        judge_binary, {'P.10', 'recip_rank'})
    res_bin = evaluator_bina.evaluate(run)
    P_10 = median([value['P_10'] for value in res_bin.values()])
    recip_rank = median([value['recip_rank']
                        for value in res_bin.values()])
    print(f'P_10 is {P_10}')
    print(f'recip_rank is {recip_rank}')

if __name__ == '__main__':
    # 1. create doc index
    # create_dataset_index_with_mapping()

    # 2. store queries index
    # index_queries_docs()

    # 3. test if success
    #res = es.get(index=INDEX_DATASET_NAME, id=1)
    # print(res)

    # (1) Lucene accurate
    # for i in range(1, 6):
    #     print('------------------------------------------------')
    #     print(f'start run {i}')
    #     lucene_acc = Lucene_accurate()
    #     run = lucene_acc.compute_aggregate_score_for_queries_docs()
    #     evaluate(run)

    # (2) Atire accurate
    # for i in range(1, 6):
    #     print('------------------------------------------------')
    #     print(f'start run {i}')
    #     atire = Atire()
    #     run = atire.compute_aggregate_score_for_queries_docs()
    #     evaluate(run)

    # test default
    run = default_es_score()
    evaluate(run)


