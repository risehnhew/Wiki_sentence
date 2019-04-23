# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:41:44 2019

@author: Administrator
"""
import json
import torch
import torch.nn as nn

import torch.optim as optim
import torchtext
import torch.nn.functional as F
import torchtext.data as data
from torchtext import datasets
import spacy
import torchtext.vocab
#try: 
#    spacy.load('en')
#
#except:
#    !python -m spacy download en
    
#path = '//TextCNN-master//qa.json'
#with open(path, 'r') as file:
#    data_set = json.load(file)

class RNN_net(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size,BATCH_SIZE, seq_len, num_layers, x_cat):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_size) # size of dictionary, size of each embedding vector
        self.batch_size = BATCH_SIZE
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.x_cat = x_cat
        self.gru = nn.GRU(embedding_size , embedding_size, num_layers=2, bias= True,dropout = 0 )
        
        self.fc1 = nn.Linear(x_cat.size(1), self.hidden_size)
        self.fc3 = nn.Linear(x_cat.size(0), num_embeddings)

        
        
    def forward(self, x, hidden):
         #input_size: (batch_size, seq_len, input_size) with batch_first=True
        
        embedded_x = self.embed(x)
        embedded_x = embedded_x.view(self.batch_size, self.seq_len, self.embedding_size)
        #packed = nn.utils.rnn.pack_padded_sequence(embedded_x, [torch.tensor(1000),torch.tensor(1000)])
        #output = F.relu(embedded_x)
        outputs, hidden = self.gru(embedded_x, hidden)
        #outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        #outputs = outputs.squeeze(0)
        Cats = self.fc1(self.x_cat).t()
        Ht = outputs @ Cats  
        
        outputs = self.fc3(Ht)
        return outputs, hidden
    def init_hidden(self):
        # initialize the hidden state
        #(num_layers * num_directions, batch, hidden_size)
        return torch.zeros( 2, self.seq_len, self.embedding_size)






