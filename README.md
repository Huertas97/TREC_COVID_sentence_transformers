# TREC_COVID_sentence_transformers

# Index
 
 * [TREC-COVID collection](#trec-covid-collection)
 
 # TREC-COVID collection 
 
 TREC-COVID [TREC-COVID](https://ir.nist.gov/covidSubmit/index.html)  is an information retrieval (IR) shared task initiated to support clinicians and clinical research during the COVID-19 pandemic. According to [[1]](#1), the basic TREC (Text REtrieval Conference) ad hoc evaluation structure provides participants with a corpus and set of topics (which they fashion into queries entered into their IR systems). Participants then submit “runs” of up to N results per topic (usually N = 1000). 
 
 In this repository the code to evaluate  [Sentence Transformers](https://www.sbert.net/index.html) models in TREC-COVID collection is available. To evaluate a model in this IR task the approach explained in [[2]](#2) is followed with slightly differeces. For the BM25 relevance score we apply the well-known BM25 Okapi algorithm, and for the sentence embeddings computing we apply bi-encoders. Bi-encoders have less accuracy than cross-encoders [REFERENCIA], but requires less computational sources and make feasible its application into real-world problems.  
 
 
 # Scripts
 
 The repository is composed of the following scripts:
 
 * The script `build_trec_covid_data.py` download the TREC-COVID data and CORD-19 documents valid for TREC-COVID round 1. 
 
 * The script `bm25_trec_covid.py` computes the relevance scores with [BM25 Okapi algorithm](https://github.com/dorianbrown/rank_bm25) between the different fields of the topic and the different facets of each document. For each topic the scores are log-normalised setting the log-base such that the highest scoring document has a value of nine. 
 
 * The script `cos_sim_trec_covid.py` computes the embeddings for the different fields of the topic and the title and abstract facets of each document and reports the semantic similarity with a cosine similarity score.
 
 In short, the final relevance score for a CORD-19 document considering a specific topic is calculated using the following formula:
 
 <img src="https://latex.codecogs.com/png.latex?%5Clarge%20%5Cpsi%20%28T_i%2C%20d%29%20%3D%20%5Clog_z%28%5Csum%5E%7Bt%5Cin%20T_%7Bi%7D%20%7D%20%5Csum%5E%7Bf%5Cin%20d%20%7DBM25%28t%2Cf%29%29%20&plus;%20%5Csum%5E%7Bt%5Cin%20T_%7Bi%7D%20%7D%20%5Csum%5E%7Bf%5Cin%20d%20%7Dcos%28e%28t%29%2C%20e%28f%29%29">

 <img src="https://latex.codecogs.com/png.latex?\inline&space;\LARGE&space;z"> represents the adjusted log-base such that the highest scoring document has a value of nine
 
<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;t&space;\in&space;T_i> represents possible fields of topic <img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;T_i> (i.e, query, question and narrative). 

<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;f&space;\in&space;d> represents possible facets of the document (i.e, abstract or title)

<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;BM25> denotes BM25 Okapi scoring algorithm

<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;e(t),&space;e(f)> represent the topic field embedding and facet embedding, respectively

<img src=https://latex.codecogs.com/png.latex?\inline&space;\large&space;cos> denotes cosine similarity


 
 
## References
<a id="1">[1]</a> 
Roberts, K., Alam, T., Bedrick, S., Demner-Fushman, D., Lo, K., Soboroff, I., … Hersh, W. R. (2020). TREC-COVID: Rationale and structure of an information retrieval shared task for COVID-19. Journal of the American Medical Informatics Association, 27(9), 1431-1436. doi: 10.1093/jamia/ocaa091

<a id="2">[2]</a> 
Nguyen, V., Rybinsk, M., Karimi, S., & Xing, Z. (2020). Searching Scientific Literature for Answers on COVID-19 Questions.
