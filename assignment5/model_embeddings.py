#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
from vocab import VocabEntry

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab: VocabEntry):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        #pad_token_idx = vocab.src['<pad>']
        #self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.char_embed_size = 50
        self.embed_size = embed_size
        self.char_embedding = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=0)
        self.cnn = CNN(char_embed_size=self.char_embed_size, word_embed_size=embed_size, kernel_size=5)
        self.highway = Highway(word_embed_size=embed_size)
        self.dropout = nn.Dropout(0.3)
        ### END YOUR CODE

    def forward(self, input: torch.Tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        batch_size, seq_len, max_word_length = input.shape[1], input.shape[0], input.shape[2]
        # print('batch size', batch_size)
        # print('max word length', max_word_length)
        #print('seq len', seq_len)

        x_char_embed = self.char_embedding(input)  # shape: (sentence_length, batch_size, max_word_length, e_char)
        #print('x_char embed shape', x_char_embed.shape)
        x_reshaped = x_char_embed.permute(0, 1, 3, 2)  # shape: (sentence_length, batch_size, e_char, max_word_length)
        #print('x_reshaped shape', x_reshaped.shape)
        x_conv = self.cnn(x_reshaped.view(-1, self.char_embed_size, max_word_length)) # shape (seq_len*batch_size, e_word)
        #print('x_conv shape', x_conv.shape)
        x_highway = self.highway(x_conv)  # shape: (batch_size*seq_len, e_word)
        #print('x_highway shape', x_highway.shape)
        x_word_embed = self.dropout(x_highway.view(seq_len, batch_size, self.embed_size))
        #print('x_word embed shape', x_word_embed.shape)

        return x_word_embed
        ### END YOUR CODE

