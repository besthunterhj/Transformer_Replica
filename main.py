import copy
import math
from typing import Callable, Tuple

import altair as alt
import pandas as pd
import torch
import torch.nn as nn
from torch import log_softmax


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture;
    and the Transformer will be implemented based on this architecture and other specific components
    """

    def __init__(self, encoder, decoder, src_embed: nn.Sequential, tgt_embed: nn.Sequential, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt: torch.Tensor, tgt_mask) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask, tgt_mask):
        """
        Take in and process masked source and target sequences
        :param src: source sequences -> [batch_size, seq_len, embedding_size]
        :param tgt: target sequences -> [batch_size, seq_len, embedding_size]
        :param src_mask:
        :param tgt_mask:
        :return: the generated sequences -> [batch_size, seq_len]
        """
        return self.decode(self.encode(src=src, src_mask=src_mask), src_mask=src_mask, tgt=tgt, tgt_mask=tgt_mask)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step
    """

    def __init__(self, d_model: int, vocab_size: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Predict the token of current step
        :param x: the output of Transformer Decoder -> [batch_size, seq_len, d_model]
        :return: the probability distribution of the prediction
        """
        return log_softmax(self.proj(x), dim=-1)


def clones(module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. (Stacked Encoding Layers and Decoding Layers)
    :param module: the layer which is used to copy n times
    :param n: the number of the stack of layer
    :return: the stacked layers
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    """
    Construct a layer normalization which can combines with residual connection
    """

    def __init__(self, size: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The process of the Layer Normalization
        :param x: the input tensor
        :return: the results of the Layer Normalization
        """

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Implement the residual connection mentioned by the original paper.
    To simplify the code, we first implement the "Norm" instead of the "Add", which is the reverse order of the paper.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size=size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Apply residual connection to any sublayer with the same size
        :param x: the input tensor
        :param sublayer: the return of the multi-head attention layer or feed forward layer
        :return: the output of current sublayer
        """

        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    The Encoder Layer consists of two sub-layers named "multi-head attention layer" and "feed forward layer"
    """

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # the encoder consists of two sub-layers, thus it needs 2 sublayer_connections
        self.sublayer_connections = clones(SublayerConnection(size=size, dropout=dropout), 2)
        # the size of Layer Normalization
        self.size = size

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        """
        The process of the encoding for Transformer model
        :param x: the input tensor, which is [batch_size, seq_len, embedding_size]
        :param mask: optional, which is usually used to implement the Masked Multi-head Attention of the Decoder
        :return: the output of the Encoder of Transformer model
        """
        # finish the first sub-layer(self-attn) of the Encoder Layer
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, mask))

        # finish the second sub-layer(FFN) of the Encoder Layer
        return self.sublayer_connections[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Core Encoder is a stack of N Encoding Layers
    """

    def __init__(self, layer: nn.Module, n: int):
        super(Encoder, self).__init__()
        self.layers = clones(module=layer, n=n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        """
        Pass the input through each Encoder Layer in turn
        :param x: the input tensor, which should be [batch_size, seq_len, embedding_size]
        :param mask: optional, which is usually used to implement the Masked Multi-head Attention of the Decoder
        :return: the output of the Encoder, which will be sent to the Decoder
        """
        # the Encoder consists of 6 encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    """
    Generic N layers decoder with masking
    """

    def __init__(self, layer, n: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask, trg_mask):
        # †he Decoder consists of 6 decoder layers
        for layer in self.layers:
            x = layer(x, memory, src_mask, trg_mask)

        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder Layer consists of 3 layers: multi-head-self-attention, multi-head-src-attention, FFN
    """

    def __init__(self, size: int, self_attn, src_attn, feed_forward, dropout: float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(size=size, dropout=dropout), 3)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask, tgt_mask):
        """
        The process of decoding of the Transformer model
        :param x: the input tensor
        :param memory: the output of the Encoder
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        m = memory

        # finish the first sublayer of Decoder(multi-head-self-attention)
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # finish the second sublayer of Decoder(multi-head-src-attention)
        x = self.sublayer_connections[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        # finish the last sublayer of Decoder(FFN)
        return self.sublayer_connections[2](x, self.feed_forward)


def subsequent_mask(size: int):
    """
    Mask out the subsequent tokens to avoid current token to attend the token from future
    ensures that the predictions for position i can depend only on the known outputs at positions less than i.
    :param size: equals to seq_len, which represents the size of the self-attention scores matrix of current sentence
    :return: a boolean variable which represents the mask value of current situation
    """
    attn_shape = (1, size, size)

    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # will return a matrix which contains boolean elements according to the judgement (subsequent_mask == 0 )
    return subsequent_mask == 0


def mask_example():
    """
    Visualize the subsequent_mask matrix
    :return: an object that records the information about the image
    """
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data).mark_rect().properties(height=250, width=250).encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None, dropout=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Scaled Dot Product Attention
    :param query: a matrix whose size should be [batch_size, h_n, seq_len, d_k]
    :param key: a matrix whose size should be [batch_size, h_n, seq_len, d_k]
    :param value: a matrix whose size should be [batch_size, h_n, seq_len, d_v]
    :param mask: Optional, it represents a symbol whether utilizing the mask mechanism
    :param dropout: Optional, it represents a symbol whether utilizing Dropout mechanism
    :return: weighted_value: [batch_size, h_n, seq_len, d_v]
    """

    d_k = query.size(-1)
    # compute the attention score by query and key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # the parameter of this function "mask" should be a tensor whose shape like "scores"
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # compute the result of softmax function
    p_attn = scores.softmax(dim=-1)

    # if the dropout function( the specific process of dropout) is sent to this function
    if dropout is not None:
        p_attn = dropout(p_attn)

    # the output of torch.matmul(p_attn, value) is the representation of sequence
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Implement the Multi-head Attention mechanism
    """

    def __init__(self, h_n: int, d_model: int, dropout: float = 0.1):
        """
        the setting of the multi-head attention mechanism
        :param h_n: the number of the heads
        :param d_model: the dimension setting of Transformer which is unified (usually is 512)
        :param dropout: the probability of the dropout
        """
        super(MultiHeadAttention, self).__init__()
        # Firstly, we guarantee the result of d_model divides h_n is an integer
        assert d_model % h_n == 0

        self.h_n = h_n
        # we assume that d_v always equals to d_k and the operator "//" output an integer instead of float by "/"
        self.d_k = d_model // h_n
        # init 4 matrices: W_Q, W_K, W_V, W_O
        self.linears = clones(nn.Linear(in_features=d_model, out_features=d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        """
        Implement the function -> MultiHead(Q, K, V)
        :param query: the Q matrix, whose size is [batch_size, seq_len, d_model]
        :param key: the K matrix, whose size is [batch_size, seq_len, d_model]
        :param value: the V matrix, whose size is [batch_size, seq_len, d_model]
        :param mask: Optional, which should be like [batch_size, seq_len]
        :return:
        """

        if mask is not None:
            # Same mask vector applied to all h heads.
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            # change the size of query, key, value to [batch_size, h_n, seq_len, d_k]
            lin(x).view(batch_size, -1, self.h_n, self.d_k).transpose(1, 2)
            # lin is current matrix of list [W_Q, W_K, W_K]; x is current matrix of [query, key, value]
            for lin, x in zip(self.linears[:-1], (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # query, key, value: [batch_size, h_n, seq_len, d_k]
        # weighted_value: [batch_size, h_n, seq_len, d_k]
        weighted_multi_head_value, self.attn = attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # weighted_value: [batch_size, seq_len, d_model]
        weighted_value = (
            weighted_multi_head_value.transpose(1, 2).contiguous().view(batch_size, -1, self.h_n * self.d_k)
        )

        # delete the query, key, value which is [batch_size, h_n, seq_len, d_k]
        del query
        del key
        del value

        # implement the linear transformation of "Linear(concat(head_1, ..., head_n), W_O)"
        return self.linears[-1](weighted_value)


class PositionWiseFeedForward(nn.Module):
    """
    Implement the FFN equation from the paper
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.w_2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The process of the FFN function
        :param x: the input tensor => [batch_size, seq_len, d_model]
        :return: the result of the FFN function, which is as the same size of the input
        """
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    """
    Differ from the vanilla Embedding, the Embedding Layers used at Transformer multiply math.sqrt(d_model)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the Position Encoding
    """

    def __init__(self, d_model: int, dropout, max_len: int = 5000):
        """
        The setting of Position Encoding
        :param d_model: the dimension of the Embedding
        :param max_len: the max length of the sequence
        :param dropout: the rate of dropout
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        # init the "pos" variable of the equation
        position = torch.arange(0, max_len).unsqueeze(1)
        # init the denominator of the equation: "10000^(2i/d_model)"
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # the position of even in the sequence is a result of sin function
        pe[:, 0::2] = torch.sin(position * div_term)
        # the position of odd in the sequence is a result of cos function
        pe[:, 1::2] = torch.cos(position * div_term)

        # let the shape of the variable "position_encoding" like [batch_size, seq_len, embedding_size]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


def make_model():
    pass

if __name__ == '__main__':
    pass
