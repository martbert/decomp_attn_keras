# Utilities related to dealing with the Quora set
import os

import pandas as pd
import numpy as np
import spacy

# Import all functions from utils
from utils.utils import *

"""Class that contains the data and methods to extract word vector based
features for training deep learning algorithms

Description.
"""
class Data:
    def __init__(self,
        sents1,
        sents2,
        targets,
        max_length=50,
        nlp_vectors=None,
        normed=True,
        pos_to_remove=['PUNCT'],
        lemmatize=False):

        self.max_length = max_length
        self.nlp_vectors = nlp_vectors
        self.normed = normed
        self.pos_to_remove = pos_to_remove
        self.lemmatize = lemmatize

        self.sents1 = sents1
        self.sents2 = sents2
        self.targets = targets

        self.min_t = np.min(targets)
        self.max_t = np.max(targets)

        # Spacy's NLP
        print('Loading NLP...')
        self.load_nlp()

        # Placeholder for features
        self.features = {}

        # Placeholder's for indices
        self.train_idx = None
        self.valid_idx = None

    def load_nlp(self):
        if self.nlp_vectors is not None:
            # Check whether its an actual path and try loading from it
            if os.path.exists(self.nlp_vectors):
                self.nlp = spacy.load('en_core_web_sm')
                load_vectors_in_lang(self.nlp, self.nlp_vectors)
            else:
                # Try loading from an installed reference
                self.nlp = spacy.load('en', vectors=self.nlp_vectors)
        else:
            self.nlp = spacy.load('en_core_web_md')

    def calc_features(self, s):
        return pad(
            get_matrix_rep(
                s, 
                self.nlp, 
                normed=self.normed,
                pos_to_remove=self.pos_to_remove,
                lemmatize=self.lemmatize
            ), 
            self.max_length
        )

    def get_sent_features(self, sent_hash):
        s,h = sent_hash
        M = self.features.get(h, None)
        if M is None:
            M = self.calc_features(s)
            self.features[h] = M
        return M

    def get_features_for_indices(self, indices):
        X1 = np.array([self.get_sent_features(self.sents1[i]) for i in indices], dtype=np.float32)
        X2 = np.array([self.get_sent_features(self.sents2[i]) for i in indices], dtype=np.float32)
        return X1,X2

    def batch_generator(self, indices_set='train', batch_size=128,
        single_pass=False, noise=0):
        
        if indices_set == 'train':
            jdx = self.train_idx.copy()
        elif indices_set == 'valid':
            jdx = self.valid_idx.copy()
        
        nb = int(jdx.shape[0] / batch_size)
        while True:
            np.random.shuffle(jdx)
            for i in range(nb):
                start = i*batch_size
                j = jdx[start:start + batch_size]
                X1, X2 = self.get_features_for_indices(j)
                t = self.targets[j]
                if noise:
                    n = np.abs(noise*np.random.randn(batch_size))
                    t += n
                    t = np.where(t > self.max_t, self.max_t, t)
                    t = np.where(t < self.min_t, self.min_t, t)
                yield ([X1, X2], t)
            if single_pass:
                break

"""Class that contains the data and methods specifically related to SNLI

Description.
"""
class SNLIData(Data):

    def __init__(self, 
        data_path=None,
        max_length=50,
        nlp_vectors=None,
        normed=False,
        pos_to_remove=['PUNCT'],
        add_questions=None,
        categorize=False
        ):
        
        self.max_length = max_length
        self.nlp_vectors = nlp_vectors
        self.pos_to_remove = pos_to_remove
        
        # Look for the SNLI data and load
        print('Reading training data...')
        pairs = pd.read_csv(data_path)

        # Make sure index is linear increasing by increments of 1
        pairs.index = list(range(pairs.shape[0]))

        # Create hash for each sentence
        print('Hashing questions...')
        pairs['hash1'] = [permanent_hash(s) for s in pairs.sentence1]
        pairs['hash2'] = [permanent_hash(s) for s in pairs.sentence2]

        # Store targets
        print('Storing targets...')
        if categorize:
            ent_to_num = {'entailment': [1,0,0], 'neutral':[0,1,0], 'contradiction':[0,0,1]}
            targets = np.array([ent_to_num[ent] for ent in pairs.gold_label])
        else:
            ent_to_num = {'entailment': 1, 'neutral':0.5, 'contradiction':0}
            targets = pairs.gold_label.map(lambda x: ent_to_num[x]).values
        targets = targets.astype(np.float64)

        # Store information relevant to features in dict for speed
        print('Storing feature relevant info...')
        sents1 = {}
        sents2 = {}
        for idx,row in pairs.iterrows():
            sents1[idx] = (row.sentence1, row.hash1)
            sents2[idx] = (row.sentence2, row.hash2)

        # Initialize Data
        Data.__init__(self, 
            sents1,
            sents2,
            targets,
            max_length=max_length,
            nlp_vectors=nlp_vectors,
            normed=normed,
            pos_to_remove=pos_to_remove,
            )

        # Initialize train/validation indices
        self.train_idx = np.where(pairs['SET'].values == 'TRAIN')[0]
        self.valid_idx = np.where(pairs['SET'].values == 'DEV')[0]

    def save_oov_to_path(self):
        oov_path,ext = os.path.splitext(self.nlp_vectors)
        oov_path = oov_path+'.oov.txt'
        save_oov(self.nlp, oov_path)

