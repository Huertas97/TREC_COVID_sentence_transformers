# TREC_COVID_sentence_transformers

# Index
 
 * [TREC-COVID collection](#trec-covid-collection)
 * [Scripts](#scripts)
 * [Metrics](#metrics)
 * [Models evaluated](#models-evaluated)
 * [How to use](#how-to-use)
 * [References](#references)
 
 # TREC-COVID collection 
 
[TREC-COVID](https://ir.nist.gov/covidSubmit/index.html)  is an information retrieval (IR) shared task initiated to support clinicians and clinical research during the COVID-19 pandemic. According to [[1]](#1), the basic TREC (Text REtrieval Conference) ad hoc evaluation structure provides participants with a corpus and set of topics (which they fashion into queries entered into their IR systems). Participants then submit “runs” of up to N results per topic (usually N = 1000). 
 
In this repository the code to evaluate  [Sentence Transformers](https://www.sbert.net/index.html) models in TREC-COVID collection is available. To evaluate a model in this IR task the approach explained in [[2]](#2) is followed with slightly differeces. For the BM25 relevance score we apply the well-known BM25 Okapi algorithm, and for the sentence embeddings computing we apply bi-encoders. Bi-encoders have less accuracy than cross-encoders [[3]](#3), but requires less computational sources and make feasible its application into real-world problems.  
 
 
[TREC-COVID](https://ir.nist.gov/covidSubmit/index.html) is an information retrieval (IR) shared task initiated to support clinicians and clinical research during the COVID-19 pandemic. According to [[1]](#1), the basic TREC (Text REtrieval Conference) ad hoc evaluation structure provides participants with a corpus and set of topics (which they fashion into queries entered into their IR systems). Participants then submit “runs” of up to N results per topic (usually N = 1000).

In this repository, the code to evaluate [Sentence Transformers](https://www.sbert.net/index.html) models in TREC-COVID collection is available. , the approach explained in [[2]](#2) is followed to evaluate a model in this IR task with slight differences. For the BM25 relevance score, we apply the well-known BM25 Okapi algorithm. For the sentence embeddings computing, we apply bi-encoders. Bi-encoders have less accuracy than cross-encoders [[3]](#3), but requires less computational sources and make feasible its application into real-world problems.
 
 # Scripts
 
 The repository is composed of the following scripts:
 
 * The script `build_trec_covid_data.py` downloads the TREC-COVID data and CORD-19 documents valid for TREC-COVID round 1. 
 
 * The script `bm25_trec_covid.py` computes the relevance scores with [BM25 Okapi algorithm](https://github.com/dorianbrown/rank_bm25) between the different fields of the topic and the different facets of each document. For each topic, the scores are log-normalised setting the log-base such that the highest scoring document has a value of nine.
 
 * The script `cos_sim_trec_covid.py` computes the embeddings for the different fields of the topic, the title and abstract facets of each document, and reports the semantic similarity with a cosine similarity score. The script `ensemble_cos_sim_trec_covid.py ` follows the same strategy, but computing the embeddings with an ensemble of different sentence transformer models. Finally, `ensemble_dim_red_cos_sim_trec_covid.py`combines the ensemble models with its repective dimentional reduction PCA. 
 
 In short, the final relevance score for a CORD-19 document considering a specific topic is calculated using the following formula:
 <p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Clarge%20%5Cpsi%20%28T_i%2C%20d%29%20%3D%20%5Clog_z%28%5Csum%5E%7Bt%5Cin%20T_%7Bi%7D%20%7D%20%5Csum%5E%7Bf%5Cin%20d%20%7DBM25%28t%2Cf%29%29%20&plus;%20%5Csum%5E%7Bt%5Cin%20T_%7Bi%7D%20%7D%20%5Csum%5E%7Bf%5Cin%20d%20%7Dcos%28e%28t%29%2C%20e%28f%29%29">
</p>

Where:
 <img src="https://latex.codecogs.com/png.latex?\inline&space;\large&space;z"> represents the adjusted log-base such that the highest scoring document has a value of nine
 
<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;t&space;\in&space;T_i> represents possible fields of topic <img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;T_i> (i.e, query, question and narrative). 

<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;f&space;\in&space;d> represents possible facets of the document (i.e, abstract or title)

<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;BM25> denotes BM25 Okapi scoring algorithm

<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;e(t),&space;e(f)> represent the topic field embedding and facet embedding, respectively

<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;cos> denotes cosine similarity

# Metrics 
## p@5
Precision is the fraction of the documents retrieved that are relevant to the user's information need. P@5 is a precision metric computed at a cut-off rank of 5. This is, considering only the top 5 results returned by the system. If the cutoff is larger than the number of docs retrieved, then it is assumed nonrelevant docs fill in the rest.  

Eg, if a method retrieves 15 docs of which 4 are relevant, then P20 is 0.2 (4/20). Precision is a very nice user oriented measure, and a good comparison number for a single topic, but it does not average well. For example, P20 has very different expected characteristics if there 300 total relevant docs for a topic as opposed to 10.

 ## map


 ## ndcg@10
The premise of  Discounted Cumulative Gain (DCG) is that highly relevant documents appearing lower in a search result list should be penalized as the graded relevance value is reduced logarithmically proportional to the position of the result. Since result set may vary in size among different queries or systems, to compare performances the normalised version is used (NDCG). NDCG divides the DCG score calculated by and ideal DCG (iDCG). The iDCG represents the perfect ranking algorithm that produces an nDCG of 1.0. 

NDCG@10 is a NDCG metric computed at a cut-off rank of 10. This is, considering only the top 10 results returned by the system.

## bpref
 The main Binary Preference (bpref) [[4]](#4) measure represents the fraction of the top R nonrelevant docs that are retrieved after each relevant doc. Put another way: when looking at the R relevant docs, and the top R nonrelevant docs, if all relevant docs are to be preferred to nonrelevant docs, bpref is the fraction of the preferences that the ranking preserves.  Bpref penalises a system if it ranks a judged nonrelevant document above a judged relevant one, and is indepedendent of how the unjudged documents are retrieved.
 



# Models evaluated
Official scores metrics for TREC-COVID task round 1 have been calculated for the following models. You can consult the ranking of top 50 official runs in this [leaderboard](https://docs.google.com/spreadsheets/d/1NOu3ZVNar_amKwFzD4tGLz3ff2yRulL3Tbg_PzUSR_U/edit#gid=1068033221):

| BM25 + Transformer-based bi-encoder model            |   p@5  | ndcg@10 |   map  |  bpref |
|------------------------------------------------------|:------:|:-------:|:------:|:------:|
| distiluse-base-multilingual-cased *                  | 0.7067 |  0.6043 | 0.2268 | 0.3964 |
| xlm-r-distilroberta-base-paraphrase-v1 * **          |  0.72  |  0.5812 | 0.2127 | 0.3854 |
| xlm-r-bert-base-nli-stsb-mean-tokens * **            | 0.6267 |  0.5354 | 0.1918 | 0.3732 |
| LaBSE *                                              |  0.72  |  0.6316 | 0.2433 | 0.4036 |
| distilbert-multilingual-nli-stsb-quora-ranking *     | 0.7267 |  0.6006 | 0.2312 | 0.3773 |
| distiluse-base-multilingual-cased + PCA              | 0.6733 |  0.5896 |  0.223 | 0.3989 |
| xlm-r-distilroberta-base-paraphrase-v1 + PCA         | 0.6533 |  0.5565 | 0.1994 | 0.3816 |
| xlm-r-bert-base-nli-stsb-mean-tokens + PCA           |  0.58  |  0.5012 | 0.1779 | 0.3671 |
| LaBSE + PCA                                          |  0.74  |   0.63  | 0.2373 | 0.4045 |
| distilbert-multilingual-nli-stsb-quora-ranking + PCA | 0.6267 |  0.5218 | 0.1839 | 0.3552 |
| Ensemble 5 models *                                  | 0.7067 |  0.5874 | 0.2165 | 0.3811 |
| Ensemble 5 models + PCA *                            |        |         |        |        |
| Ensemble 2   best models (best from STSb) **         | 0.6667 |  0.5425 | 0.2002 |  0.378 |
| Ensemble 2 best models + PCA (best from STSb) **     |        |         |        |        |
| Ensemble 3 best models (best from TREC-COVID)        |  0.74  |  0.6023 | 0.2325 | 0.3779 |
| Ensemble 3 best models + PCA (best from TREC-COVID)  |        |         |        |        |
| BERT-base                                            | 0.7067 |  0.6071 | 0.2238 | 0.3801 |
| RoBERTa                                              |  0.68  |  0.5969 | 0.2239 |  0.379 |
| clinicalcovid-bert-nli                               |  0.74  |  0.6303 | 0.2309 | 0.4074 |
| scibert-nli                                          |  0.68  |  0.5861 | 0.2037 | 0.3781 |
| biobert-nli                                          |   0.7  |  0.5923 | 0.2103 | 0.3902 |


| BM25 + Sentence Transformer multilingual models |   p@5  | ndcg@10 |   map  |  bpref |
|:-----------------------------------------------:|:------:|:-------:|:------:|:------:|
| distiluse-base-multilingual-cased               | 0.7067 |  0.6043 | 0.2268 | 0.3964 |
| xlm-r-distilroberta-base-paraphrase-v1          |  0.72  |  0.5812 | 0.2127 | 0.3854 |
| xlm-r-bert-base-nli-stsb-mean-tokens            | 0.6267 |  0.5354 | 0.1918 | 0.3732 |
| LaBSE                                           |  0.72  |  0.6316 | 0.2433 | 0.4036 |
| distilbert-multilingual-nli-stsb-quora-ranking  | 0.7267 |  0.6006 | 0.2312 | 0.3773 |

| BM25 + Multilingual models with dimensional reduction |   p@5  | ndcg@10 |   map  |  bpref |
|:-----------------------------------------------------:|:------:|:-------:|:------:|:------:|
| distiluse-base-multilingual-cased + PCA               | 0.6733 |  0.5896 |  0.223 | 0.3989 |
| xlm-r-distilroberta-base-paraphrase-v1 + PCA          | 0.6533 |  0.5565 | 0.1994 | 0.3816 |
| xlm-r-bert-base-nli-stsb-mean-tokens + PCA            |  0.58  |  0.5012 | 0.1779 | 0.3671 |
| LaBSE + PCA                                           |  0.74  |   0.63  | 0.2373 | 0.4045 |
| distilbert-multilingual-nli-stsb-quora-ranking + PCA  | 0.6267 |  0.5218 | 0.1839 | 0.3552 |


|   BM25 + Transfromer-based model (no task adapted)  |   p@5  | ndcg@10 |   map  |  bpref |
|:---------------------------------------------------:|:------:|:-------:|:------:|:------:|
| BERT-base                                           | 0.7067 |  0.6071 | 0.2238 | 0.3801 |
| RoBERTa                                             |  0.68  |  0.5969 | 0.2239 |  0.379 |

|      BM25 + Biomedical transfromer-based model      |  p@5 | ndcg@10 |   map  |  bpref |
|:---------------------------------------------------:|:----:|:-------:|:------:|:------:|
| clinicalcovid-bert-nli                              | 0.74 |  0.6303 | 0.2309 | 0.4074 |
| scibert-nli                                         | 0.68 |  0.5861 | 0.2037 | 0.3781 |
| biobert-nli                                         |  0.7 |  0.5923 | 0.2103 | 0.3902 |
  

 

# How to use 

 
# References
<a id="1">[1]</a> 
Roberts, K., Alam, T., Bedrick, S., Demner-Fushman, D., Lo, K., Soboroff, I., … Hersh, W. R. (2020). TREC-COVID: Rationale and structure of an information retrieval shared task for COVID-19. Journal of the American Medical Informatics Association, 27(9), 1431-1436. doi: 10.1093/jamia/ocaa091

<a id="2">[2]</a> 
Nguyen, V., Rybinsk, M., Karimi, S., & Xing, Z. (2020). Searching Scientific Literature for Answers on COVID-19 Questions.

<a id="3">[3]</a> 
Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv:1908.10084 [cs]. Recuperado de http://arxiv.org/abs/1908.10084

<a id="4">[4</a> 
Buckley, C., & Voorhees, E. M. (2004). Retrieval evaluation with incomplete information. Proceedings of the 27th Annual International Conference on Research and Development in Information Retrieval  - SIGIR ’04, 25. Sheffield, United Kingdom: ACM Press. doi: 10.1145/1008992.1009000
