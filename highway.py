#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn

class Highway(nn.Module):
    """
    Implements a one-layer ReLU highway network, that can take batched inputs (where the
    first dimension is the batch-size dimension
    """
    def __init__(self, word_emb_size):
        """
        Initializes a 1-layer ReLU network, given the word embedding (output tensor) size.
        """
        super(Highway, self).__init__()
        self.linear_proj = nn.Linear(word_emb_size, word_emb_size)
        self.linear_gate = nn.Linear(word_emb_size, word_emb_size)

    def forward(self, x):
        """
        Computes the output of the 1-layer ReLU highway network given a batched input
        tensor x
        Takes in a tensor of shape (batch_size, num_channels)
        Outputs a tensor of shape (batch_size, num_channels)
        """
        x_proj = self.linear_proj(x).clamp(min=0)
        x_gate = torch.sigmoid(self.linear_gate(x))
        return x_gate * x_proj + (1 - x_gate) * x

### END YOUR CODE 

