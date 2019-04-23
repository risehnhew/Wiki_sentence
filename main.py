# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from config import Config
from model import TextCNN
import numpy as np
from numpy import array
from torch.utils import data
import json
from gensim.models import FastText 
import argparse
import time

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--gpu', type=int, default= 0)
parser.add_argument('--out_channel', type=int, default=2)
parser.add_argument('--label_num', type=int, default=2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_data_path', type= str, default='data/Sentences.json')

args = parser.parse_args()


torch.manual_seed(args.seed)

#if torch.cuda.is_available():
#    device = 'cuda'
#else: device = 'cpu'

device = 'cpu'

# -----data processing--------
with open(args.train_data_path, 'r') as f1:

    sentence = json.load(f1)
    
    sentence2 = dict()
    sentence2['wiki_elements'] = sentence['wiki_elements'][0:1000] # simply choosing 1000 samples
    sentence2['wiki_sentences_length'] = sentence['wiki_sentences_length'][0:1000]
    sentence2['wiki_sentences'] = sentence['wiki_sentences'][0:1000]
    sentence2['wiki_titles'] = sentence['wiki_titles'][0:1000]
    
    max_len = max(sentence2['wiki_sentences_length'])
    
    for i in range(len(sentence2['wiki_sentences_length'])):
        if sentence2['wiki_sentences_length'][i] < max(sentence2['wiki_sentences_length']):
            length = sentence2['wiki_sentences_length'][i]                   
            ext_pad = (max_len - length) * ['<PAD>'] # padding
            sentence2['wiki_sentences'][i].extend(ext_pad)
    data2 = sentence2['wiki_sentences']

    label3 = sentence2['wiki_titles']
    #--------------------
    
    label2 = []
    dict_lab = dict()
    i0 = 0

    for ii, keys in enumerate(label3): # numeralize labels
        if keys not in dict_lab.keys():
            dict_lab[keys] = i0
            i0+=1

    for i3, keys2 in enumerate(label3):
        label2.append(dict_lab[keys2]) 


    
    embedding_model = FastText(data2, min_count=1, size = 100) # embedding   
    data3= []
    for sentence2 in data2:
        data3.append(embedding_model.wv[sentence2])# word embedding
    class_num = torch.tensor(label2).unique().numel()
       
    label2 = array(label2)
    label2 = label2.reshape(len(label2), 1)
    
    #-------------------
    label_t = torch.tensor(label2,dtype=torch.long).squeeze().to(device)
    data_t = torch.tensor(data3).to(device)

#--------------------CNN part-------------------------------------------
config = Config(sentence_max_size=50,
                batch_size=args.batch_size,
                word_num=11000,
                label_num=class_num,
                learning_rate=args.lr,
                cuda=args.gpu,
                epoch=args.epoch,
                out_channel=args.out_channel)
class TextDataset(data.Dataset):

    def __init__(self, label2, data2):

        self.label = label2.squeeze()
        self.data = data2

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

training_set = TextDataset(label_t, data_t)
training_iter = data.DataLoader(dataset=training_set,
                                batch_size=config.batch_size,
                                drop_last=True
                                )

model = TextCNN(config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

count = 0
loss_sum = 0
# Train the model


start = time.time()
for epoch in range(config.epoch):

    optimizer.zero_grad()
    loss_sum = 0
    for data, label in training_iter:

        #input_data = embeds(data)
        input_data = data.unsqueeze(1)
        out, x_cat = model(input_data)
        x_cat = x_cat.squeeze(1)
        x_cat = x_cat.squeeze(1)
        loss = criterion(out, label)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss_sum))
    # save the model in every epoch
    #model.save('checkpoints/epoch{}.ckpt'.format(epoch))
end = time.time()

print('time is: ',  end - start)
    
#----------------------------------------------------------
#----------------------RNN part--------------------------------

import json
import numpy as np
import h5py
import torch

import torch.nn as nn

import torch.optim as optim
import torchtext
import torchtext.data as data
from torchtext import datasets
import spacy
import torchtext.vocab
#import RNN_model 
import itertools
import torchtext
from RNN_model import RNN_net
from gensim.models import FastText 


glove = torchtext.vocab.GloVe(name='6B', dim=50)

print(f'There are {len(glove.itos)} words in the vocabulary')
pretrained_embeddings = glove.vectors

reddit_path = 'data/Aligned-Dataset/reddit.h5'
wikipedia_path = 'data/Aligned-Dataset/wikipedia.h5'
dictionary_path = 'data/Aligned-Dataset/dictionary.json'

reddit = h5py.File(reddit_path, 'r')
wikipedia = h5py.File(wikipedia_path, 'r')

BATCH_SIZE = 1 
device = torch.device('cpu')
num_embeddings = 400000
embedding_size = 50
num_layers =2
seq_len = 1000
#input_lengths = [sentence_len, BATCH_SIZE, embedding_size]
hidden_size = embedding_size


with open(dictionary_path, 'r') as f:
    #dictionary = json.load(f, 'utf-8')
    dictionary = json.load(f)
    id2word = dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = dictionary['word2id']
    f.close()
index = 1
dataset = 'train'
PAD_id = word2id['<PAD>']

           
if index < len(reddit[dataset]):
        i = 0
        sequence = ''
        sequence1 = []
        sequences = []
        while reddit[dataset][index][i + 1] != word2id['<PAD>']:
            if reddit[dataset][index][i] == word2id['<end>'] or reddit[dataset][index][i] == word2id['<eot>']: 
                sequence1.append(reddit[dataset][index][i].astype(int))
                sequences.append(sequence1)# end of the sentence
                
                sequence = sequence + id2word[reddit[dataset][index][i]] + '\n'
                sequence1 = []#
                
            else:
                
                sequence1.append(reddit[dataset][index][i].astype(int))
                sequence = sequence + id2word[reddit[dataset][index][i]] + ' ' # NOT end of the sentence
            i += 1
        sequence = sequence + id2word[reddit[dataset][index][i]]
        sentences = []
        qa = sequences[2:]
        
        
def extractSentencePairs(sequences):
    qa_pairs = []
#    for conversation in conversations:
        # Iterate over all the lines of the conversation
    for i in range(len(sequences) - 1):  # We ignore the last line (no answer for it)
        inputLine = sequences[i]
        while len(inputLine)< 1000:
            inputLine = inputLine + [PAD_id] # Padding
        targetLine = sequences[i+1]
        while len(targetLine)< 1000:
            targetLine = targetLine + [PAD_id] # Padding
        # Filter wrong samples (if one of the lists is empty)
        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])
    return qa_pairs

train_data = extractSentencePairs(qa)


    
model = RNN_net(num_embeddings = num_embeddings, embedding_size =embedding_size,
                hidden_size = hidden_size,BATCH_SIZE=BATCH_SIZE, seq_len=seq_len,num_layers = num_layers, x_cat=x_cat)
model.embed.weight.data.copy_(pretrained_embeddings)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()


for epoch in range(10):
    loss_sum = 0 
    optimizer.zero_grad()
    for i , data in enumerate(train_data):
        
        train_d = torch.tensor(data[0])
        target_d = torch.tensor(data[1])
        model_hidden = model.init_hidden()
        outputs, hidden  = model(train_d, model_hidden)
        
        loss = criterion(outputs.squeeze(0), target_d)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    
    print('Loss in epoch {} is {}'.format(epoch, loss_sum))