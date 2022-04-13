import math
import torch
import torch.nn as nn

"""
Transformer works in parallel
Positional Encoder will help the model to take benefit from words position in sentences
Works based on Original Paper Attention is All you need and PyTorch Doc
Used Sine and Cosine for different positions
Max length is the longest sentence in datasets
Dimention models is the embedding layers shape
"""
class PositionalEncoding(nn.Module):

    def __init__(self, d_model=250, max_len=143):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return x


class TransEncoder(nn.Module):

    def __init__(self, hparams):    
        super(TransEncoder, self).__init__()

        self.pos_encoder = PositionalEncoding()

        # Loads weights from previosuly generated weights
        # But it will updates the weights during training
        self.lemmas_embedding = nn.Embedding.from_pretrained(hparams.embedding_weights['lemmas'], freeze=False)
        self.pos_tags_embedding = nn.Embedding.from_pretrained(hparams.embedding_weights['pos_tags'], freeze=False)
        self.predicate_embedding = nn.Embedding.from_pretrained(hparams.embedding_weights['predicates'], freeze=False)
        self.predicate_flag_embedding = nn.Embedding.from_pretrained(hparams.embedding_weights['predicate_indicator'],
                                                                     freeze=False)
        self.lemmas_flag_embedding = nn.Embedding.from_pretrained(hparams.embedding_weights['lemmas_indicator'],
                                                                  freeze=False)

        encoder_layers = nn.TransformerEncoderLayer(hparams.dim_emb,
                                                    hparams.num_heads,
                                                    hparams.dim_feedforward,
                                                    hparams.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, hparams.nlayers)

        # Create two seperate linear layers 
        # First one is for binary labels
        # Second one is for original labels - 36 in total
        self.classifier_s3 = nn.Linear(hparams.dim_emb, hparams.num_classes_s3)
        self.classifier_s4 = nn.Linear(hparams.dim_emb, hparams.num_classes_s4)

    def forward(self, src):
        lemmas = src['lemmas']
        pos_tags = src['pos_tags']
        predicates = src['predicates']
        predicates_indicator = src['predicate_indicator']
        lemmas_indicator = src['lemmas_indicator']

        lemmas_emb = self.lemmas_embedding(lemmas)
        pos_emb = self.pos_tags_embedding(pos_tags)
        pred_emb = self.predicate_embedding(predicates)
        pred_flag_emb = self.predicate_flag_embedding(predicates_indicator)
        lemmas_flag_emb = self.lemmas_flag_embedding(lemmas_indicator)


        # Concatenate all 5 features' embedding weights
        # Each one has 50 dim, so 250 in total
        embeddings = torch.cat((lemmas_emb, pos_emb, pred_emb, pred_flag_emb, lemmas_flag_emb), -1)
        embeddings = torch.transpose(embeddings, 0, 1)

        # Add positional embedding
        # the positional weights sums with the input embedding, 
        # So the output size would be the same as input
        embeddings = self.pos_encoder(embeddings)

        o = self.transformer_encoder(embeddings)

        output_s3 = self.classifier_s3(o)
        output_s4 = self.classifier_s4(o)   
        output_s3 = torch.transpose(output_s3, 0, 1)
        output_s4 = torch.transpose(output_s4, 0, 1)

        return output_s3, output_s4


class HParams():
    dim_feedforward = 1024
    dim_emb = 250
    dropout = 0.2
    nlayers = 6
    num_heads = 10
    num_classes_s3 = 4
    num_classes_s4 = None
    embedding_weights = None



