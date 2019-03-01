#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Implements a one-layer max-pooling CNN, that can take batched inputs (where the first
    dimension is the batch-size dimension
    """
    def __init__(self, char_embed_size, word_embed_size, kernel_size=5):
        """
        Initializes a 1-layer max-pool CNN, given the word embedding (output tensor) size.
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(char_embed_size, word_embed_size, kernel_size)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        """
        Computes the output of the 1-layer max-pool CNN given a batched input tensor x
        Takes in a tensor of shape (batch_size, in_channels, seq_length)
        Outputs a tensor of shape (batch_size, out_channels)
        """
        return self.maxpool(nn.functional.relu(self.conv(x))).squeeze()

### END YOUR CODE

