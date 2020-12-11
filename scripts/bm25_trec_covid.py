# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:09:18 2020

@author: alvar
"""



import spacy
import scispacy
import en_core_sci_sm
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
import os
import sys
import math
import pickle
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger() 

from optparse import OptionParser

# *********** COMMAND-LINE OPTIONS ***********
# Process command-line options
parser = OptionParser(add_help_option=False)

# General options
parser.add_option('-d', '--data', type="string", help='Path to TREC-COVID parsed data')
parser.add_option('-f', '--fulltext', action='store_true', default=False, help='Include fulltext corpus for BM25 scoring')
parser.add_option('-a', '--abstract', action='store_true', default=False, help='Include abstract corpus for BM25 scoring')
parser.add_option('-t', '--title', action='store_true', default=False, help='Include titles corpus for BM25 scoring')
parser.add_option('-h', '--help', action='store_true', help='Show this help message and exit.')

(options, args) = parser.parse_args()
def print_usage():
    print("""
Usage:

    python bm25_trec_covid.py [options] 

Options:
    -d, --data              Path to TREC-COVID parsed data
    -f, --fulltext          Bool: Include fulltext corpus for BM25 scoring
    -a, --abstract          Bool: Include abstract corpus for BM25 scoring  
    -t, --title             Bool: Include titles corpus for BM25 scoring  
    -h, --help              Help information

At least one of -f, -a or -t must be selected
Example:
    python bm25_trec_covid.py -f -a -t --data ./trec_covid_data/df_docs.pkl""")
    sys.exit()
    
if options.help or not options.data:
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

########### TOKENIZE FOR BM25 ###########


# BM25 tokenizer
def BM25_tokenizer(text):
  tokens = nlp.tokenizer(text.lower().replace("\n", " "))
  # remove puntuation
  tokens = [str(t) for t in tokens if not t.is_punct]
  return tokens


########### SCALE SCORES ###########

# Apply the log scale to each score
def scale_score(score, base):
  if score <= 0:
    score = 0.0000000000001
  return math.log(score, base)

# Calculate the base term for scaling the max score to 9
def base2scale(arr_per_topic, top_val):
  max_sc = np.max(arr_per_topic)
  base = max_sc**(1/top_val)
  return base


def log_top_scale(arr_per_topic, top_val = 9):
  base = base2scale(arr_per_topic, top_val)
  return np.array([scale_score(score, base) for score in arr_per_topic])



######################################################
####################### CORPUS #######################
######################################################

# # SciSpacy model to tokenize text
print("-------- Loading scispacy en_core_sci_sm model --------")
nlp = en_core_sci_sm.load(disable=['ner', 'tagger'])
nlp.max_length = 2000000


# # Corpus
print("-------- Building corpus --------")
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
# ####################### TOPICS #######################
# ######################################################

# Topic parsing
# Parse the xml file with the topics for the round
print("-------- Extracting topics --------")
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

# ######################################################
# ####################### Scores #######################
# ######################################################

corpus_score_bm25 = np.zeros( ( len(corpus_list), len(list(topics.keys())), len(corpus_list[0]) ) )
# for corpus_idx, (name_corpus, corpus) in enumerate(zip(name_corpus_list, tqdm(corpus_list, desc="Corpus") )): 
for corpus_idx, (name_corpus, corpus) in enumerate(tqdm(zip(name_corpus_list, corpus_list), total = len(corpus_list), desc="Corpus", position=0) ): 
  # Adding corpus to BM25
  print("-------- Adding {} corpus to BM25 --------".format(name_corpus))
  tokenized_corpus = [BM25_tokenizer(c) for c in tqdm(corpus, desc="Tokenized", position=0)] # tokenize corpus with Scispacy 
  bm25 = BM25Okapi(tokenized_corpus)
  
  # Extracting text from each topic
  print("-------- {}: BM25 scores for each topic --------".format(name_corpus))
  topics_score = np.zeros(  ( len(list(topics.keys())), len(corpus) )  )
  for topic_idx, (n_topic, topic_data) in enumerate(tqdm(topics.items(), desc = "Topic", position=0)):
    query = topic_data["query"]
    question = topic_data["question"]
    narrative = topic_data["narrative"]
    
    # Tokenize it for BM25
    tokenized_query = BM25_tokenizer(query.lower())
    tokenized_question = BM25_tokenizer(question.lower())
    tokenized_narrative = BM25_tokenizer(narrative.lower())

    # get scores
    query_bm25_scores = bm25.get_scores(tokenized_query)
    question_bm25_scores = bm25.get_scores(tokenized_question)
    narrative_bm25_scores = bm25.get_scores(tokenized_narrative)
    
    # sum scores for each type of topics text
    topic_score = query_bm25_scores + question_bm25_scores + narrative_bm25_scores
    topics_score[topic_idx] = topic_score
  
  # Save the result  for each corpus
  corpus_score_bm25[corpus_idx] = topics_score

# Summation of topics scores for all corpus 
logger.info("-------- Summation BM25 scores for all corpus --------")
corpus_score_bm25_sum =  np.sum(corpus_score_bm25,axis=0)

logger.info("-------- Scaling scores (max score = 9) --------")
scaled_rank_BM25sc = np.apply_along_axis(log_top_scale, 1, corpus_score_bm25_sum)

logger.info("-------- Saving BM25 results in ./results/df_BM25_sc.pkl --------")
os.makedirs("./results", exist_ok=True)
pickle.dump( scaled_rank_BM25sc, open( "./results/df_BM25_sc.pkl", "wb" ) )