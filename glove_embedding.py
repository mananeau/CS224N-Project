#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
glove_embedding.py : create word2id (dict) and glove embeddings (numpy array) files.
Guoqin Ma <sebsk@stanford.edu>
"""

import os
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_folder", type=str)
parser.add_argument("--output_folder", type=str)
args = parser.parse_args()

os.chdir(args.embedding_folder)

with open('glove.twitter.27B.200d.txt') as f:
    words = f.readlines()

word2id = dict()
embeddings = []

for i, w in enumerate(words):
    w_list = w.split(' ')
    word2id.update({w_list[0]: i})
    embeddings.append([float(j) for j in w_list[1:]])

embeddings = np.array(embeddings)
os.chdir(args.output_folder)
pickle.dump(word2id, open('glove_word2id', 'wb'))
np.save('glove_embeddings', embeddings)


