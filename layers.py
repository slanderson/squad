"""Assortment of layers for use in models.py.

Author:
    Spenser Anderson (aspenser@stanford.edu)
    Adapted from code by Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

import pdb

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        char_cnn (bool): Whether or not to use character-based CNN embeddings
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob, char_cnn=False,
                 char_size=50):
        super(Embedding, self).__init__()

        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1) + char_size, 
                              hidden_size, 
                              bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)
        self.char_emb = CNNEmbedding(char_size, char_vectors.shape[0],
                                     char_embed_size=char_vectors.shape[1],
                                     drop_prob=drop_prob) if char_cnn else None

    def forward(self, x, xc):
        c_emb = self.char_emb(xc.permute(1, 0, 2)).permute(1, 0, 2) if self.char_emb else None
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        stacked_emb = torch.cat((c_emb, emb), dim=2) if self.char_emb else emb
        emb = F.dropout(stacked_emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x

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

class CNNEmbedding(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, num_chars, padding_idx=0, char_embed_size=50, kernel=5,
                 drop_prob=0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        """
        super(CNNEmbedding, self).__init__()
        self.e_char = char_embed_size
        self.embed_size = embed_size
        self.char_embed = nn.Embedding(num_chars, char_embed_size, 
                                       padding_idx=padding_idx)
        self.cnn = CNN(char_embed_size, embed_size, kernel)
        self.highway = HighwayEncoder(1, embed_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        sent_len, batch_size, m_word = input.shape[0], input.shape[1], input.shape[2]
        embed = self.char_embed(input)
        sentence_batch_reshape = embed.reshape(sent_len * batch_size, m_word, self.e_char).permute(0, 2, 1)
        xconv_out = self.cnn(sentence_batch_reshape)
        x_word_emb = self.dropout(self.highway(xconv_out))
        return x_word_emb.reshape(sent_len, batch_size, self.embed_size)

class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
        use_lstm (bool): Use LSTM (as opposed to GRU) for RNN.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.,
                 use_lstm=False):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.) if use_lstm else\
                  nn.GRU(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x

class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class RNetAttention(nn.Module):
    """Passage-query attention in the style of RNet

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.1,
                 device=torch.device('cpu')):
        super(RNetAttention, self).__init__()
        self.p_linear = nn.Linear(input_size, hidden_size)
        self.q_linear = nn.Linear(input_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.gate_linear = nn.Linear(2*input_size, 2*input_size)
        self.gru = nn.GRUCell(input_size=2*input_size,
                              hidden_size=hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size, 1))
        nn.init.xavier_uniform_(self.v)
        self.dropout = nn.Dropout(drop_prob)
        self.device = device
        self.hidden_size = hidden_size
        

    def forward(self, u_p, u_q, p_mask, q_mask):
        batch_size, p_len, vec_size = u_p.size()
        _, q_len, _ = u_q.size()
        q_mask = q_mask.view(batch_size, q_len, 1)  # (batch_size, c_len, 1)
        v = torch.zeros(batch_size, self.hidden_size, device=self.device)
        S_q = self.q_linear(u_q)
        att_out = torch.zeros(batch_size, p_len, self.hidden_size, device=self.device)
        for i in range(p_len):
            S = torch.tanh(S_q + 
                           self.p_linear(u_p[:,i,:]).unsqueeze(1) +
                           self.v_linear(v).unsqueeze(1)).matmul(self.v)
            c = u_q.permute(0, 2, 1).matmul(masked_softmax(S, q_mask, dim=1)).squeeze()
            inp = torch.cat((c, u_p[:,i,:]), dim=1) 
            inp = torch.sigmoid(self.gate_linear(inp)) * inp
            v = self.gru(inp, v)
            att_out[:,i,:] = v
            del S, c, inp

        return att_out

class SelfAttention(nn.Module):
    """Self-attention in the style of RNet

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.1, use_lstm=False,
                 device=torch.device('cpu')):
        super(SelfAttention, self).__init__()
        self.own_linear = nn.Linear(input_size, hidden_size)
        self.comp_linear = nn.Linear(input_size, hidden_size)
        self.gate_linear = nn.Linear(2*input_size, 2*input_size)
        self.rnn = RNNEncoder(input_size=2*input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              drop_prob=drop_prob,
                              use_lstm=use_lstm)
        self.v = nn.Parameter(torch.zeros(hidden_size, 1))
        nn.init.xavier_uniform_(self.v)
        self.dropout = nn.Dropout(drop_prob)
        self.device = device

    def forward(self, v, lengths, p_mask):
        batch_size, p_len, vec_size = v.size()
        C = torch.zeros(*v.size(), device=self.device)
        p_mask = p_mask.view(batch_size, p_len, 1)  # (batch_size, c_len, 1)
        S_v = self.comp_linear(v)
        for i in range(p_len):
            S = torch.tanh(S_v + self.own_linear(v[:, i, :]).unsqueeze(1)).matmul(self.v)
            C[:, i, :]= v.permute(0, 2, 1).matmul(masked_softmax(S, p_mask, dim=1)).squeeze()
            del S

        inp = torch.cat((v, C), dim=2)
        inp = torch.sigmoid(self.gate_linear(inp)) * inp
        h = self.rnn(inp, lengths)
        return h

class ModelingLayer(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.1, use_lstm=False):
        super(ModelingLayer, self).__init__()
        self.rnn1 = RNNEncoder(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              drop_prob=drop_prob,
                              use_lstm=use_lstm)

    def forward(self, x, lengths):
        v = self.rnn1(x, lengths)
        return v

class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, att_size, mod_size, drop_prob, use_lstm=False):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(att_size, 1)
        self.mod_linear_1 = nn.Linear(mod_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob,
                              use_lstm=use_lstm)

        self.att_linear_2 = nn.Linear(att_size, 1)
        self.mod_linear_2 = nn.Linear(mod_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
