#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, Embedding
import torch.nn.functional as F
from vocab import VocabEntry
from typing import Tuple


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab:VocabEntry=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super().__init__()
        self.charDecoder = LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.decoderCharEmb = Embedding(len(target_vocab.char2id), char_embedding_size,
                                        padding_idx=target_vocab.char2id['<pad>'])
        self.char_output_projection = Linear(hidden_size, len(target_vocab.char2id))
        self.target_vocab = target_vocab
        ### END YOUR CODE

    def forward(self, input: torch.Tensor, dec_hidden = None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input_embed = self.decoderCharEmb(input)  # shape: (length, batch, self.hidden_size)
        outputs, (h_n, c_n) = self.charDecoder(input_embed, dec_hidden)  # shape of outputs: (length, batch, self.hidden_size)
        s_t = self.char_output_projection(outputs)  # shape: (length, batch, self.vocab_size)
        return s_t, (h_n, c_n)
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        s_t, (h_n, c_n) = self.forward(char_sequence[:-1], dec_hidden)
        #print('s_t shape', s_t.shape)
        #print('char_sequence shape', char_sequence.shape)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        loss = loss_fn(s_t.view(-1, len(self.target_vocab.char2id)), char_sequence[1:].contiguous().view(-1))
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]

        current_char = [self.target_vocab.char2id['{']] * batch_size  # len: (batch_size, )
        decodedWords = ['{'] * batch_size  # len: (batch_size,)

        #print('current char is', current_char)
        current_char_tensor = torch.tensor(current_char, device=device)  # shape: (batch_size,)

        #print('current char tensor shape', current_char_tensor.shape)

        h_prev, c_prev = initialStates

        for t in range(max_length):
            _, (h_new, c_new) = self.forward(current_char_tensor.unsqueeze(0),
                                                 (h_prev, c_prev))  # shape: (1, batch, hidden_size)

            s = self.char_output_projection(h_new.squeeze(0))  # shape: (batch, self.vocab_size)

            p = F.log_softmax(s, dim=1)  # shape: (batch, self.vocab_size)
            current_char_tensor = torch.argmax(p, dim=1)  # shape: (batch,)

            for i in range(batch_size):
                decodedWords[i] += self.target_vocab.id2char[current_char_tensor[i].item()]

            h_prev = h_new
            c_prev = c_new

        for i in range(batch_size):
            decodedWords[i] = decodedWords[i][1:]
            decodedWords[i] = decodedWords[i].partition('}')[0]

        return decodedWords

        ### END YOUR CODE

