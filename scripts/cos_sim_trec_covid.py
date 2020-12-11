# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:23:40 2020

@author: alvar
"""

from sentence_transformers import SentenceTransformer, util
import torch
import spacy
import scispacy
import en_core_sci_sm
import numpy as np
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
import pandas as pd
import os
import pickle
import logging
import sys
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()
from optparse import OptionParser

# *********** COMMAND-LINE OPTIONS ***********
# Process command-line options
parser = OptionParser(add_help_option=False)


# General options
parser.add_option('-d', '--data', type="string", help='Name of Transformer-based model from https://huggingface.co/pricing')
parser.add_option('-m', '--model', type="string", help='Path to TREC-COVID parsed data')
parser.add_option('-h', '--help', action='store_true', help='Show this help message and exit.')
parser.add_option('-f', '--fulltext', action='store_true', default=False, help='Include fulltext corpus for BM25 scoring')
parser.add_option('-a', '--abstract', action='store_true', default=False, help='Include abstract corpus for BM25 scoring')
parser.add_option('-t', '--title', action='store_true', default=False, help='Include titles corpus for BM25 scoring')
parser.add_option('-b', '--batch', type="int", default= 100, help='Batch size')

(options, args) = parser.parse_args()
def print_usage():
    print("""
Usage:

    python cos_sim_trec_covid.py [options] 

Options:
    -d, --data              Path to TREC-COVID parsed data
    -m, --model             Name of Transformer-based model from https://huggingface.co/pricing
    -f, --fulltext          Bool: Include fulltext corpus for BM25 scoring
    -a, --abstract          Bool: Include abstract corpus for BM25 scoring  
    -t, --title             Bool: Include titles corpus for BM25 scoring  
    -b, --batch             Batch size


Example:
    python ./scripts/cos_sim_trec_covid.py -b 1000 -t -a -f --data ./trec_covid_data/df_docs.pkl --model distiluse-base-multilingual-cased""")
    sys.exit()
    
if options.help or not options.data or not options.model:
    print_usage()
if not options.fulltext and not options.abstract and not options.title:
    print_usage()
    
######################################################
######################    DATA    ####################
######################################################

df_docs = pd.read_pickle(options.data)

######################################################
###################### FUNCTIONS #####################
######################################################

########### TOKENIZE AND COMPUTE CORPUS EMBEDDINGS ###########

# Tokenize text
def split_sentences(text):
  sentences = [sent.text.strip() for sent in nlp(text).sents]
  return sentences

# Computing text embedding averaging sentence embeddings
def compute_abs_emb(abstract, model):
  sentences = split_sentences(abstract)
  sent_emb_matrix = model.encode(sentences, convert_to_tensor = True, show_progress_bar=False)
  avg_abs_emb = torch.mean(sent_emb_matrix, 0, False)
  return avg_abs_emb



######################################################
####################### CORPUS #######################
######################################################

# # SciSpacy model to tokenize text
print("-------- Loading scispacy en_core_sci_sm model --------")
nlp = en_core_sci_sm.load(disable=['ner', 'tagger'])
nlp.max_length = 2000000

# Sentence Transformer model
logger.info("-------- Loading SentenceTransformer model --------")
embedder = SentenceTransformer(options.model)
dim = embedder.get_sentence_embedding_dimension()

# # Corpus
logger.info("-------- Building corpus --------")
df_docs.title = df_docs.title.fillna("")
df_docs.abstract = df_docs.abstract.fillna("")
df_docs.fulltext = df_docs.fulltext.fillna("")

corpus_list = []
name_corpus_list = []
if options.fulltext:
    fulltext_corpus = df_docs.fulltext.to_list()     
    corpus_list.append(fulltext_corpus)
    name_corpus_list.append("fulltext")
if options.abstract:
    abstract_corpus = df_docs.abstract.to_list()
    corpus_list.append(abstract_corpus)
    name_corpus_list.append("abstract")
if options.title:
    title_corpus = df_docs.title.to_list()
    corpus_list.append(title_corpus)
    name_corpus_list.append("title")

# ######################################################
# ####################### TOPICS ########################
# ######################################################

# Topic parsing
# Parse the xml file with the topics for the round
logger.info("-------- Extracting topics --------")
topics = {}
root = ET.parse("./trec_covid_data/topics-rnd1.xml").getroot()
for topic in root.findall("topic"):
    # create dictionary with number id of the topic as key
    topic_number = topic.attrib["number"]
    topics[topic_number] = {}
    for query in topic.findall("query"):
        topics[topic_number]["query"] = query.text
    for question in topic.findall("question"):
        topics[topic_number]["question"] = question.text        
    for narrative in topic.findall("narrative"):
        topics[topic_number]["narrative"] = narrative.text



batch_size = options.batch
corpus_score_batches = []
# corpus_list = []
for i in tqdm(range(0, len(corpus_list[0]), batch_size), desc = "Batches", position =0):
  corpus_batch_list = [c[i:i+batch_size] for c in corpus_list]
  # ft_batch_c = fulltext_corpus[i:i+batch_size]
  # abs_batch_c = abstract_corpus[i:i+batch_size]
  # t_batch_c = title_corpus[i:i+batch_size]



  ######################################################
  # Score for each corpus
  ######################################################
  print("-------- Computing scores --------")
  # name_corpus_list = ["fulltext", "abstract", "title"]
  # corpus_list = [ft_batch_c, abs_batch_c, t_batch_c]
  corpus_score_batch = np.zeros( ( len(corpus_list), len(list(topics.keys())), len(corpus_batch_list[0]) ) )

  for corpus_idx, (name_corpus, corpus) in enumerate(zip(name_corpus_list, corpus_batch_list)): 
    # Computing corpus embeddings
    # print("-------- Computing {} embeddings --------".format(name_corpus))
    # print("'\r{0}".format(idx), end='')
    # print("'\r-------- Computing {} embeddings --------".format(name_corpus), end="")
    corpus_embeddings = torch.zeros(size = (len(corpus), dim), dtype=torch.float32)
    for i, t in enumerate( tqdm(corpus, total = len(corpus), desc = name_corpus + " embeddings", position = 0, leave=False) ):
      if t != "" or len(t) != 0:
        corpus_embeddings[i] = compute_abs_emb(t, embedder)
      else:
        corpus_embeddings[i] = embedder.encode(t, convert_to_tensor=True, show_progress_bar=False)


    
    ###############################################################
    # Score for each text from topic (query, question, narrative)
    ##############################################################
    
    # Extracting text from each topic
    topics_score = np.zeros(  ( len(list(topics.keys())), len(corpus) )  )
    # logger.info("-------- Computing BM25 and cosine similarity scores --------".format(name_corpus))
    for topic_idx, (n_topic, topic_data) in enumerate(tqdm(topics.items(), desc = "Topic", position =0, leave = False)):
      query = topic_data["query"]
      question = topic_data["question"]
      narrative = topic_data["narrative"]


      # Corpus vs Topic Query scores
      query_embedding = embedder.encode(query, convert_to_tensor=True, show_progress_bar=False)
      cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
      query_cos_scores = cos_scores.cpu()

      # Corpus vs Topic Question scores
      question_embedding = embedder.encode(question, convert_to_tensor=True, show_progress_bar=False)
      cos_scores = util.pytorch_cos_sim(question_embedding, corpus_embeddings)[0]
      question_cos_scores = cos_scores.cpu()


      # Corpus vs Topic Narrative scores
      narrative_embedding = embedder.encode(narrative, convert_to_tensor=True, show_progress_bar=False)
      cos_scores = util.pytorch_cos_sim(narrative_embedding, corpus_embeddings)[0]
      narrative_cos_scores = cos_scores.cpu()

      # Score for a topìc
      topic_score = query_cos_scores + question_cos_scores  + narrative_cos_scores 
      topics_score[topic_idx] = topic_score
    
    # Saving score for all topics for a corpus
    corpus_score_batch[corpus_idx] = topics_score
  
  # Saving result for each batch
  corpus_score_batches.append(corpus_score_batch)


# Summation of topics scores for all corpus 
logger.info("-------- Summation cosine similairty scores for all batches --------")
corpus_score = np.concatenate(corpus_score_batches, axis=2)
logger.info("-------- Summation cosine similairty scores for all corpus --------")
corpus_score_sum = np.sum(corpus_score,axis=0)

model_name = options.model.split("/")[-1]

logger.info("-------- Saving results in ./results/df_cos_sim_sc_{}.pkl --------".format(model_name))
os.makedirs("./results", exist_ok=True)
pickle.dump( corpus_score_sum, open( "./results/df_cos_sim_sc_" + model_name +".pkl", "wb" ) )      
