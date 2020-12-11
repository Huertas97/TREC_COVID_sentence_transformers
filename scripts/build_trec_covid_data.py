# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 23:39:07 2020

@author: alvar
"""

import logging
from sentence_transformers import SentenceTransformer, LoggingHandler, models, util
import io
import os
import tarfile
from tqdm import tqdm
import pandas as pd
import json
import datetime

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

# ROUND 1 TREC-COVID
# URLs
download_url_docids = "https://ir.nist.gov/covidSubmit/data/docids-rnd1.txt"
download_url_topics = "https://ir.nist.gov/covidSubmit/data/topics-rnd1.xml"
download_url_qrels = "https://ir.nist.gov/covidSubmit/data/qrels-rnd1.txt"
download_url_cord = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-04-10.tar.gz"
urls = [download_url_docids, download_url_topics, download_url_qrels, download_url_cord]

# PATHs
path_docids = "./trec_covid_data/docids-rnd1.txt"
path_topics = "./trec_covid_data/topics-rnd1.xml"
path_qrels = "./trec_covid_data/qrels-rnd1.txt"
path_cord = "./trec_covid_data/cord-19_2020-04-10.tar.gz"
paths = [path_docids, path_topics, path_qrels, path_cord]


# Download the TREC-COVID data
for url, path in zip(urls, paths):
  logger.info("------ Downloading {} ------".format(path.split("/")[-1]))
  util.http_get(url, path)

# Uncompress files (json files for fulltext)
class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        self.progress = tqdm(unit="B", total=self._total_size, unit_scale=True)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        self.progress.update(size)
        return io.FileIO.read(self, size)

logger.info("------ Uncompressing cord-19_2020-04-10.tar.gz ------")
obj = ProgressFileObject("./trec_covid_data/cord-19_2020-04-10.tar.gz")
tar = tarfile.open(fileobj=obj)
tar.extractall()
obj.progress.close()
tar.close()


for f in ["noncomm_use_subset", "custom_license", "comm_use_subset", "biorxiv_medrxiv"]:
  logger.info("------ Uncompressing ./2020-04-10/{}.tar.gz ------".format(f))
  obj = ProgressFileObject("./2020-04-10/{}.tar.gz".format(f))
  tar = tarfile.open(fileobj=obj)
  tar.extractall(path="./2020-04-10/")
  obj.progress.close()
  tar.close()


# COVID Parser to get documents title, abstract and fulltext from CORD-19 data
class Parser:
    @classmethod
    def parse_document(cls, *args, **kwargs):
        raise NotImplementedError()


class CovidParser(Parser):
    is_str = lambda k: isinstance(k, str)
    data_path = "./2020-04-10/"

    @classmethod
    def parse(cls, row):

        document = {'id': row['cord_uid'],
                    'title': row['title']}

        if cls.is_str(row['full_text_file']):
            path = None
            if row['has_pmc_xml_parse']:
                path = cls.data_path + row['full_text_file'] + \
                       '/pmc_json/' + row['pmcid'].split(';')[ 0] +\
                       '.xml.json'

            elif row['has_pdf_parse']:
                path = cls.data_path + row['full_text_file'] +\
                       '/pdf_json/' + row['sha'].split(';')[0] +\
                       '.json'

            if path:
                content = json.load(open(path))
                fulltext = '\n'.join([p['text'] for p in content['body_text']])
                document['fulltext'] = fulltext
        else:
            document['fulltext'] = ''

        if cls.is_str(row['abstract']):
            document['abstract']=row['abstract']
        else:
            document['abstract']=''
        if cls.is_str(row['publish_time']):
            date = row['publish_time']
            len_is_4 = len(row['publish_time']) == 4
            document['date'] = f'{date}-01-01' if len_is_4 else date

        return document

class CovidParserNew(CovidParser):
    @classmethod
    def parse(cls, row):
        if not cls.is_str(row['title']):
          row["title"] = ""
        document = {'id': row['cord_uid'],
                    'title': row['title']}

        path = None
        if cls.is_str(row['pdf_json_files']):
            data_row = row['pdf_json_files'].split(';')[0].strip()
            path = cls.data_path + data_row

        elif cls.is_str(row['pmc_json_files']):
            data_row = row['pmc_json_files'].split(';')[0].strip()
            path = cls.data_path + data_row

        if path:
            content = json.load(open(path))
            fulltext = '\n'.join([p['text'] for p in content['body_text']])
            document['fulltext'] = fulltext

        else:
            document['fulltext'] = ''

        if cls.is_str(row['abstract']):
            document['abstract']=row['abstract']
        else:
            document['abstract']=''
        if cls.is_str(row['publish_time']):
            date = row['publish_time']
            len_is_4 = len(row['publish_time']) == 4
            document['date'] = f'{date}-01-01' if len_is_4 else date

        return document

# Prepare CORD-19 data frame for TREC-COVID task
import pandas as pd
metafile = "./2020-04-10/metadata.csv"
df_metadata = pd.read_csv(metafile, index_col=None)

# Clean the data frame: only allowed ids, remove duplicated, drop date before 31-12-19
#############################################
cord_uids = []
with open("./trec_covid_data/docids-rnd1.txt", "r") as f:
  for line in f.readlines():
    cord_uids.append(line.strip())

# Only allowed cord_uids
df_metadata = df_metadata[ df_metadata.cord_uid.isin(cord_uids)]
# Drop cord_uids duplicated
df_metadata = df_metadata.drop_duplicates(subset=["cord_uid"], ignore_index=True)
# Drop documents with a publication date before 31-12-19 (COVID's first case reported)
df_metadata.publish_time = pd.to_datetime(df_metadata.publish_time)
df_metadata = df_metadata[df_metadata['publish_time'] >= datetime.datetime(2019, 12, 31)]
df_metadata.reset_index(drop =True, inplace=True)
#############################################


# Create the TREC-COVID data frame
df_docs = pd.DataFrame()
for _, row in tqdm(df_metadata.iterrows(), total=len(df_metadata.index), desc = "CORD documents extracted"):
  doc = CovidParser.parse(row)
  df_docs = df_docs.append(doc, ignore_index=True)

# Save results
logger.info("------ Saving parsed CORD-19 documents ------")
df_docs.title = df_docs.title.fillna("")
df_docs.abstract = df_docs.abstract.fillna("")
df_docs.fulltext = df_docs.fulltext.fillna("")
df_docs.to_pickle("./trec_covid_data/df_docs.pkl")
logger.info("Saved in trec_covid_data/df_docs.pkl")
logger.info("------ Finished ------")