from turtle import position
import ir_datasets
import numpy as np
from ir_datasets.datasets.clinicaltrials import ClinicalTrialsDocs, ClinicalTrialsDoc
from elasticsearch import Elasticsearch
    
es = Elasticsearch()
dataset: ClinicalTrialsDocs = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")
# (hyper) parameters
N = dataset.docs_count()
RETRIEVE_SIZE = 10000
k1 = 1.2
b = 0.75

INDEX_DATASET_NAME = "clinical-trial-track"
INDEX_QUERIES_NAME = "ctt-queries"
DATASET_MAPPING_SIMPLE = {
  "settings": {
        "index":{
                "number_of_shards":1,
                "number_of_replicas":1
                }
            },
  "mappings":{
      "properties":{
          "doc_id":{
              "type": "text"
          },
          "summary":{
              "type": "text",
              "fielddata":"true",
              "term_vector":"with_positions_offsets_payloads",
              "store":"true",
              "analyzer":"whitespace"
          }
        }      
    }
}
DATASET_MAPPING = {
  "settings": {
        "index":{
                "number_of_shards":1,
                "number_of_replicas":1
                }
            },
  "mappings":{
      "properties":{
          "doc_id":{
              "type": "text"
          },
          "title":{
              "type": "text",
              "fielddata":"true",
              "term_vector":"with_positions_offsets_payloads",
              "store":"true",
              "analyzer":"whitespace"
          },
          "condition":{
              "type": "text",
              "fielddata":"true",
              "term_vector":"with_positions_offsets_payloads",
              "store":"true",
              "analyzer":"whitespace"
          },
          "summary":{
              "type": "text",
              "fielddata":"true",
              "term_vector":"with_positions_offsets_payloads",
              "store":"true",
              "analyzer":"whitespace"
          },
          "detaiiled_description":{
              "type": "text",
              "fielddata":"true",
              "term_vector":"with_positions_offsets_payloads",
              "store":"true",
              "analyzer":"whitespace"
          },
          "eligibility":{
              "type": "text",
              "fielddata":"true",
              "term_vector":"with_positions_offsets_payloads",
              "store":"true",
              "analyzer":"whitespace"
            }
        }      
    }
}

def create_dataset_index_with_mapping():
    es.indices.create(index=INDEX_DATASET_NAME, body=DATASET_MAPPING_SIMPLE)
    index_dataset_docs()

def index_queries_docs():
    import pandas as pd
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

def search_by_query(query_id=1):
    query_result = es.get(index=INDEX_QUERIES_NAME, id=query_id)
    query_text = query_result['_source']['query']
    query_body = define_query_body(query_text)
    doc_list = es.search(index=INDEX_DATASET_NAME, body=query_body)
    return doc_list['hits']['hits']

def index_dataset_docs(simple=True):
    for idx, doc in enumerate(dataset.docs_iter()): # namedtuple<doc_id, title, condition, summary, detailed_description, eligibility>
        if not simple:
            doc = define_dataset_fields(doc.doc_id, doc.title, doc.condition, doc.summary, doc.detailed_description, doc.eligibility)
        else:
            doc = define_dataset_fields(doc.doc_id, summary=doc.summary, simple=True)
        es.index(index=INDEX_DATASET_NAME, body=doc, id=idx)

def delete_index(index='clinical-trial-track'):
    es.indices.delete(index=index)

def get_indices_name():
    indices = es.indices.get(index='*')
    return indices.keys()

def get_splitted_query(query_id=1):
    query_text = es.get(index=INDEX_QUERIES_NAME, id=query_id)['_source']['query']
    return query_text.split(' ')

def get_field_statistics(index=INDEX_DATASET_NAME, doc_id=1, field = 'summary', terms=False):
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
    doc = es.get(index=INDEX_DATASET_NAME, id=doc_id)['_source']
    text: str = doc[field]
    splitted_text = text.split()
    return len(splitted_text)

def get_doc_ids_by_query(query_id=1):
    docs = search_by_query(query_id)
    doc_ids = [doc['_id'] for doc in docs]
    return doc_ids

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
    
    def compute_aggregate_score_for_query_docs(self, query_id, field='summary'):
        splitted_query = get_splitted_query(query_id=query_id)
        doc_ids = get_doc_ids_by_query(query_id=query_id)
        scores_list = []
        doc_count = 0
        for j in range(25):
            mterm_vectors = es.mtermvectors(index=INDEX_DATASET_NAME, ids=doc_ids[400*j:400*(j+1)], term_statistics=True)["docs"]
            for i in range(400):
                doc_id = mterm_vectors[i]["_id"]
                terms = mterm_vectors[i]["term_vectors"][field]["terms"]
                scores = self.compute_aggregate_score_for_query_doc(splitted_query, doc_id, terms)
                scores_list.append(scores)
                doc_count += 1
                if doc_count % 200 == 0:
                    print("Query_Id: {}, Count:{}, Doc_Id:{}, Score:{}".format(query_id, doc_count, doc_id, scores))
        print('Max score: {}, min score: {}'.format(max(scores_list), min(scores_list)))
            # for idx, doc_id in enumerate(doc_ids):
            #     terms = mterm_vectors["docs"][int(doc_id)]["term_vectors"][field]["terms"]
            #     scores = self.compute_aggregate_score_for_query_doc(splitted_query, doc_id, terms)
            #     scores_list.append(scores)
            #     print(idx, doc_id, scores)
    
    def compute_aggregate_score_for_queries_docs(self):
        for query_id in range(1, 76):
            self.compute_aggregate_score_for_query_docs(query_id=query_id)
            print('----------------------------------------------')
            print()

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
        scores_list = []
        doc_count = 0
        for j in range(25):
            mterm_vectors = es.mtermvectors(index=INDEX_DATASET_NAME, ids=doc_ids[400*j:400*(j+1)], term_statistics=True)["docs"]
            for i in range(400):
                doc_id = mterm_vectors[i]["_id"]
                terms = mterm_vectors[i]["term_vectors"][field]["terms"]
                scores = self.compute_aggregate_score_for_query_doc(splitted_query, doc_id, terms)
                scores_list.append(scores)
                doc_count += 1
                if doc_count % 200 == 0:
                    print("Query_Id: {}, Docs_Count:{}, Doc_Id:{}, Score:{}".format(query_id, doc_count, doc_id, scores))
        print('Max score: {}, min score: {}'.format(max(scores_list), min(scores_list)))
    
    def compute_aggregate_score_for_queries_docs(self):
        for query_id in range(1, 76):
            self.compute_aggregate_score_for_query_docs(query_id=query_id)
            print('----------------------------------------------')
            print()
        
# (1) Test on Lucene accurate
lucene_acc = Lucene_accurate()
scores_list = lucene_acc.compute_aggregate_score_for_queries_docs()

# (2) Test on Atire
# atire = Atire()
# scores_list = atire.compute_aggregate_score_for_queries_docs()