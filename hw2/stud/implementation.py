import json
import random
import pickle
import logging


import torch
import torch.nn as nn
import torch.optim as optim

from model import Model

from stud.preprocessing import single_predicate_converter, create_batches
from stud.transformer_model import TransEncoder, HParams
from stud.trainer import Trainer
from stud.merger_decoder import sentence_merger, sentence_decoder


def build_model_34(device: str) -> Model:
    return StudentModel()

def build_model_234(device: str) -> Model:
    raise NotImplementedError


def build_model_1234(device: str) -> Model:
    raise NotImplementedError


class StudentModel(Model):

    PAD_TOKEN = '<pad>'

    """
    Load Vocabs Dictionary

    vocab2ids = {
        'lemmas': {},
        'pos_tags': {},
        'predicate': {},
        'roles': {},
        'predicate_indicator': {},
        'lemmas_indicator': {},
        'bi_roles': {}
    }
    """
    with open('model/vocab2ids_v1', 'rb') as fp:
        vocab2ids = pickle.load(fp)

    """
    Load Embedding weights

    embedding_weights = {
        'lemmas':{},
        'pos_tags':{},
        'predicates': {},
        'predicate_indicator': {},
        'lemmas_indicator': {}
    }
        """
    with open('model/embedding_weights_v2', 'rb') as fp:
        embedding_weights = pickle.load(fp)

    """
    Create id to label(roles) dictionary
    Makes decoding easier
    """
    id2class = {v: k for k, v in vocab2ids['roles'].items()}


    """
    Load the trained Model
    set number of output classes to the number unique labels (36)
    """
    params = HParams()
    params.num_classes_s4 = len(vocab2ids['roles'])
    params.embedding_weights = embedding_weights

    srl_model = TransEncoder(hparams=params)
    srl_model.load_state_dict(torch.load('model/Transformer_Glove_2Step_30E_F1024_L6_H10_Em250_D2_relu.pth', map_location=torch.device('cpu')))
    srl_model.eval()


    """
    Define trainer class
    We used two different loss functions one for binary labels, one for multi labels
    We set reduction to 'none' so the loss function applys no operations on the loss values
    We needed original loss values to perform our approach for improving the results
    """
    srl_trainer = Trainer(
        model=srl_model,
        loss_function_s3=nn.CrossEntropyLoss(ignore_index=vocab2ids['bi_roles'][PAD_TOKEN], reduction='none'),
        loss_function_s4=nn.CrossEntropyLoss(ignore_index=vocab2ids['roles'][PAD_TOKEN], reduction='none'),
        optimizer=optim.Adam(srl_model.parameters()),
    )



    def predict(self, sentences):
        
        # Convert multi-predicate sentences to multiple one predicate sentences
        test_single_predicate_sentences = single_predicate_converter(sentences)

        # If the sentence has at least one predicate
        if len(test_single_predicate_sentences) != 0:

            # create batches in respect to the number of predicates each sentence has
            test_batches = create_batches(test_single_predicate_sentences, len(test_single_predicate_sentences), self.vocab2ids)

            # Prepare the dataset batches
            # Specifically seperate 'position predicate' feature from other
            # Store each predicate prediction in 'test_predictions' list
            test_predictions = []
            test_predPos = []
            for sentence in test_batches:
                tokens = {
                    'lemmas': sentence['lemmas'],
                    'pos_tags': sentence['pos_tags'],
                    'predicates': sentence['predicate'],
                    'predicate_indicator': sentence['predicate_indicator'],
                    'lemmas_indicator': sentence['lemmas_indicator']

                }
                others = {
                    'position_predicate': sentence['position_predicate']
                }

                predicts = self.srl_trainer.predict(tokens)
                test_predictions.append(predicts)
                test_predPos.append(others)

            # Sentence Merger
            roles_prediction = {}
            roles_prediction = sentence_merger(test_predictions, test_predPos)

            # Sentence Decoder
            decoded_predictions = sentence_decoder(roles_prediction, self.id2class)

            return decoded_predictions
        
        # if the sentence has no predicates, return an empty dictionary
        else:
            empty_pred = {'roles': {}}
            return empty_pred
