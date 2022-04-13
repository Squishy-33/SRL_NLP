import os

import numpy as np
import torch
import torch.nn as nn


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


"""
This function convert multi-predicate sentences to multiple single predicate sentences 
The function for Train and Dev datasets is totally different and exits in the original Notebook
This function altered for single test sentence input according the code written in the 'evaluate.py' file

Output:
new format for each sentence would be:
    new_sentences[indx] = {

            'position_predicate': pred_indx,
            'lemmas': sentence['lemmas'],
            'pos_tags': sentence['pos_tags'],
            'predicate': new_pred,
            'predicate_indicator': predicate_indicator,
            'lemmas_indicator' : lemmas_indicator
        })

"""
def single_predicate_converter(sentence):

    new_sentences = []
    predicate_indexes = []
    sentence_predicates = sentence['predicates']

    # Extract predicate indexes 
    for indx, item in enumerate(sentence_predicates):
        if item != '_':
            predicate_indexes.append(indx)

    
    # Build new predicate
    # Each time considere one predicate and masks others
    # For example: _ , EAT/BITE , _ , DRINK --> _ , _ , _ , DRINK & _ , EAT/BITE , _ , _
    #  
    for pred_indx in predicate_indexes:
        new_pred = ['_']*len(sentence_predicates)
        new_pred[pred_indx] = sentence_predicates[pred_indx]

        # create predicate indicator feature
        # for example: the cat ate the fish --> 0, 0, 1, 0, 0
        predicate_indicator = [0]*len(sentence_predicates)
        predicate_indicator[pred_indx] = 1 

        # create word indicator feature in respect to predicate position
        # for example: the cat ate the fish --> -2, -1, 0, 1, 2
        lemmas_indicator = [0]*len(sentence_predicates)
        for i, x in enumerate(predicate_indicator):
            lemmas_indicator[i] = i - pred_indx


        new_sentences.append({
            'position_predicate': pred_indx,
            'lemmas': sentence['lemmas'],
            'pos_tags': sentence['pos_tags'],
            'predicate': new_pred,
            'predicate_indicator': predicate_indicator,
            'lemmas_indicator' : lemmas_indicator
        })

    return new_sentences


# Checks whether the word inside test sentences exists in the dictionary or not
# If not, replace it with unknown token
def vocab_checker(vocab2id_k, item):
    if item in vocab2id_k:
        return vocab2id_k[item]
    else:
        return vocab2id_k[UNK_TOKEN]



"""
We don't use Torch DataLoader
Input: for each sentence we have different features (depends on Test / Train-Dev)
Instead of 
          sentence1 = {'feature1', 'feature2',...}
          sentence2 = {'feature1', 'feature2',...}

We create: 
          feature1 = {'sentence1', 'sentence2',...} 
          feature2 = {'sentence1', 'sentence2',...}

The original batch function in the Notebook is more complicated
The 'create_batches' function in the notebook also handels padding and length


"""
def create_batches(dataset, batch_size, vocab2ids):
    batched_dataset = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        keyed_batch = {}
        for k in dataset[0].keys():
            x = [sentence[k] for sentence in batch]
            if k == 'position_predicate':
                keyed_batch[k] = x
                continue
            x = [xx + [PAD_TOKEN]*33 for xx in x]
            keyed_batch[k] = torch.tensor([[vocab_checker(vocab2ids[k], xxx) for xxx in xx] for xx in x])
        batched_dataset.append(keyed_batch)

    return batched_dataset
