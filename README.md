# TREC_COVID_sentence_transformers

# Index
 
 * [TREC-COVID collection](#trec-covid-collection)
 * [Scripts](#scripts)
 * [Metrics](#metrics)
 * [Models evaluated](#models-evalauted)
 * [How to use](#how-to-use)
 * [References](#references)
 
 # TREC-COVID collection 
 
 TREC-COVID [TREC-COVID](https://ir.nist.gov/covidSubmit/index.html)  is an information retrieval (IR) shared task initiated to support clinicians and clinical research during the COVID-19 pandemic. According to [[1]](#1), the basic TREC (Text REtrieval Conference) ad hoc evaluation structure provides participants with a corpus and set of topics (which they fashion into queries entered into their IR systems). Participants then submit “runs” of up to N results per topic (usually N = 1000). 
 
 In this repository the code to evaluate  [Sentence Transformers](https://www.sbert.net/index.html) models in TREC-COVID collection is available. To evaluate a model in this IR task the approach explained in [[2]](#2) is followed with slightly differeces. For the BM25 relevance score we apply the well-known BM25 Okapi algorithm, and for the sentence embeddings computing we apply bi-encoders. Bi-encoders have less accuracy than cross-encoders [REFERENCIA], but requires less computational sources and make feasible its application into real-world problems.  
 
 
 # Scripts
 
 The repository is composed of the following scripts:
 
 * The script `build_trec_covid_data.py` download the TREC-COVID data and CORD-19 documents valid for TREC-COVID round 1. 
 
 * The script `bm25_trec_covid.py` computes the relevance scores with [BM25 Okapi algorithm](https://github.com/dorianbrown/rank_bm25) between the different fields of the topic and the different facets of each document. For each topic the scores are log-normalised setting the log-base such that the highest scoring document has a value of nine. 
 
 * The script `cos_sim_trec_covid.py` computes the embeddings for the different fields of the topic and the title and abstract facets of each document and reports the semantic similarity with a cosine similarity score.
 
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

## bpref
 Main binary preference measure.\n\
    Fraction of the top R nonrelevant docs that are retrieved after each\n\
    relevant doc. Put another way: when looking at the R relevant docs, and\n\
    the top R nonrelevant docs, if all relevant docs are to be preferred to\n\
    nonrelevant docs, bpref is the fraction of the preferences that the\n\
    ranking preserves.\n\
    Cite: 'Retrieval Evaluation with Incomplete Information', Chris Buckley\n\
    and Ellen Voorhees. In Proceedings of 27th SIGIR, 2004.\n",
 
 ## ndcg@10
The premise of  Discounted Cumulative Gain (DCG) is that highly relevant documents appearing lower in a search result list should be penalized as the graded relevance value is reduced logarithmically proportional to the position of the result. Since result set may vary in size among different queries or systems, to compare performances the normalised version is used (NDCG). NDCG divides the DCG score calculated by and ideal DCG (iDCG). The iDCG represents the perfect ranking algorithm that produces an nDCG of 1.0. 

NDCG@10 is a NDCG metric computed at a cut-off rank of 10. This is, considering only the top 10 results returned by the system.


# Models evaluated
Official scores for TREC-COVID task round 1 have been calculated for the following models:

| BM25 + Transformer-based bi-encoder model            |   p@5  | ndcg@10 |   map  |  bpref |
|------------------------------------------------------|:------:|:-------:|:------:|:------:|
| distiluse-base-multilingual-cased                    | 0.7067 |  0.6043 | 0.2268 | 0.3964 |
| xlm-r-distilroberta-base-paraphrase-v1               |  0.72  |  0.5812 | 0.2127 | 0.3854 |
| xlm-r-bert-base-nli-stsb-mean-tokens                 | 0.6267 |  0.5354 | 0.1918 | 0.3732 |
| LaBSE                                                |  0.72  |  0.6316 | 0.2433 | 0.4036 |
| distilbert-multilingual-nli-stsb-quora-ranking       | 0.7267 |  0.6006 | 0.2312 | 0.3773 |
| distiluse-base-multilingual-cased + PCA              |        |         |        |        |
| xlm-r-distilroberta-base-paraphrase-v1 + PCA         |        |         |        |        |
| xlm-r-bert-base-nli-stsb-mean-tokens + PCA           |        |         |        |        |
| LaBSE + PCA                                          |        |         |        |        |
| distilbert-multilingual-nli-stsb-quora-ranking + PCA |        |         |        |        |
| Ensemble 5  models                                   |        |         |        |        |
| Ensemble 5 models + PCA                              |        |         |        |        |
| Ensemble 2  best models                              |        |         |        |        |
| Ensemble 2 best models + PCA                         |        |         |        |        |
| BERT-base                                            | 0.7067 |  0.6071 | 0.2238 | 0.3801 |
| RoBERTa                                              |  0.68  |  0.5969 | 0.2239 |  0.379 |
| clinicalcovid-bert-nli                               |  0.74  |  0.6303 | 0.2309 | 0.4074 |
| scibert-nli                                          |  0.68  |  0.5861 | 0.2037 | 0.3781 |
| biobert-nli                                          |   0.7  |  0.5923 | 0.2103 | 0.3902 |




    

  

 

# How to use 

 
# References
<a id="1">[1]</a> 
Roberts, K., Alam, T., Bedrick, S., Demner-Fushman, D., Lo, K., Soboroff, I., … Hersh, W. R. (2020). TREC-COVID: Rationale and structure of an information retrieval shared task for COVID-19. Journal of the American Medical Informatics Association, 27(9), 1431-1436. doi: 10.1093/jamia/ocaa091

<a id="2">[2]</a> 
Nguyen, V., Rybinsk, M., Karimi, S., & Xing, Z. (2020). Searching Scientific Literature for Answers on COVID-19 Questions.
