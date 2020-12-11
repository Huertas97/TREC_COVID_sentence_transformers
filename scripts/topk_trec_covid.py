# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:23:40 2020

@author: alvar
"""
import scipy.stats as ss
import numpy as np
import pandas as pd
import os
import sys
import logging
import pickle
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
parser.add_option('-p', '--path', type="string", default = "./",  help='Path to scores files')
parser.add_option('-f', '--file', type="string", help='Names of files used to sum and rank')
parser.add_option('-o', '--output', type="string", default = "topk_scores", help='Output file name')
parser.add_option('-h', '--help', action='store_true', help='Show this help message and exit.')
parser.add_option('-k', '--topk', type="int", default=1000, help='Top k scores will be retrieved from the scores files')


(options, args) = parser.parse_args()
def print_usage():
    print("""
Usage:

    python topk_trec_covid.py [options] 

Options:
    -d, --data              Path to TREC-COVID parsed data
    -m, --model             Name of Transformer-based model from https://huggingface.co/pricing
    -f, --file              Names of files used to sum and rank
    -o, --output            Output file name
    -h. --help              Show this help message and exit
    -k. --topk              Top k scores will be retrieved from the scores files
    -p, --path              Path to scores files


Example:
    python ./scripts/topk_trec_covid.py --data ./trec_covid_data/df_docs.pkl -p ./results -f df_BM25_sc.pkl -o bm25_topk_sc""")
    sys.exit()
    
if options.help or not options.file or not options.data:
    print_usage()


######################################################
#################    FUNCTIONS    ####################
######################################################
def read_pickle(pkl_name):
   return  pickle.load( open( pkl_name, "rb" )) 

# Get indexes for top k  scaled scores
def top_k(array, k=1000):
  top_idx = np.argpartition(array, -k)[-k:]
  return top_idx

# Calculate the ranking position over the top k scores
def calculate_ranking(array):
  return len(array) - ss.rankdata(array).astype("int") +1


######################################################
######################    DATA    ####################
######################################################

# TREC-COVID dataframe  with parsed docs
df_docs = pd.read_pickle(options.data)

# Files with the scores calculated
name_files = options.file.split(",")
sc_files = [ read_pickle(os.path.join(options.path, f)) for f in name_files]

# Summation of scores
scores = np.sum( (sc_files) , axis=0)

logger.info("-------- Retrieving top 1000 scores for each topic --------")
k = options.topk
# top k index

top_k_idx = np.apply_along_axis(top_k, 1, scores, k=k)

#top k scores
top_k_sc = np.array([ arr[idx] for arr, idx in zip(scores, top_k_idx) ] )
top_k_ranks = np.apply_along_axis(calculate_ranking, 1, top_k_sc)
top_k_sc = np.concatenate(top_k_sc)
top_k_ranks = np.concatenate(top_k_ranks)



# top k topics ids
n_topics = scores.shape[0]
top_k_topic_id = np.repeat(range(1, n_topics+1), k)

# top k cord_uids
top_k_ids = []
for idx in top_k_idx:
  top_k_ids += df_docs.loc[idx, "id"].to_list()

logger.info("-------- Saving results in ./results/{} --------".format(options.output))
df_results = pd.DataFrame({"topicid":top_k_topic_id,
              "Q0": ["Q0"]* top_k_topic_id.shape[0],
              "docid": top_k_ids,
              "rank": top_k_ranks,
              "score": top_k_sc,
              "run-tag": ["TFM"]* top_k_topic_id.shape[0]
              })
os.makedirs("./results", exist_ok=True)
df_results.to_csv('./results/' + options.output +'.txt', header=None, index=None, sep='\t', mode='w')

logger.info("-------- Finished --------")