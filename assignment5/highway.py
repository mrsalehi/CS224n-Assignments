#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super().__init__()
        self.proj_layer = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.gate_layer = nn.Linear(word_embed_size, word_embed_size, bias=True)

    def forward(self, input: torch.Tensor):
        """
        :param input: tensor with the shape (batch_size, e_word)
        :return:
        """
        x_proj = F.relu(self.proj_layer(input))
        x_gate = torch.sigmoid(self.gate_layer(input))
        x_highway = x_gate * x_proj + (1 - x_gate) * input
        return x_highway

### END YOUR CODE 

