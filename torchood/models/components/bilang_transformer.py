import math

import torch
import torch.nn as nn

"""
d_model : Dimantionality of model [512]
seq_len : sequence length
d_ff : twice of input d_model dimension coming in
voceb_szie : Total number of words in input
query (torch.Tensor): The query tensor of shape (batch_size, seq_len, d_model).
key (torch.Tensor): The key tensor of shape (batch_size, seq_len, d_model).
value (torch.Tensor): The value tensor of shape (batch_size, seq_len, d_model).
mask (torch.Tensor): The mask tensor of shape (batch_size, seq_len) or None.
dropout (nn.Dropout): The dropout layer.

"""


# A simple Layer Normalization code, with bias=False
# USE -> The LayerNormalization class implements the layer normalization technique in neural networks. It normalizes the input tensor along the last dimension (hidden size) using learnable parameters.
# The need to rewrite, because torch default laternorm does not let us keep bias as False
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Alpha a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))  # Bias a learnable parameter

    def forward(self, x):
        """Normalizes the input tensor along the last dimension using layer normalization.

        Applies learnable scaling and shifting to the normalized tensor.
        """
        # X -> (batch,seq_len,hidden_size)
        # Keeping the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # EPS(epsilon value) to prevent divide by 0 and for numerical stability.
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# The Feedforward Network
# USE -> In encoder and decoder blocks, a module in a transformer model that applies a feed-forward neural network to each position in the input sequence independently.
# Takes combination of head output and convert that in new output (Mixing)
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Applies a linear transformation to the input tensor using two linear layers with ReLU
        activation in between.

        Adds dropout regularization to the output of the first linear layer. Returns the output
        tensor after the second linear layer.
        """
        # X -> (batch,seq_len,d_model) -> (batch,seq_len,d_ff) -> (batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# Input Embedding
# USE -> convert input sequence into Embeddings
# The InputEmbedding class is responsible for creating an embedding layer for the input data in the Transformer model. It takes the input dimension (d_model) and the vocabulary size (vocab_size) as parameters and initializes an embedding layer using the nn.Embedding module.
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """Create an embedding layer for the input data in the Transformer model.

        Scale the embedding values by math.sqrt(self.d_model) for numerical stability.
        """
        # x-> (batch,seq_len) -> (batch,seq_len,de_model)
        # match.sqrt according to model (After Numerical research and for numerical stability)
        return self.embedding(x) * math.sqrt(self.d_model)


# Position Embedding
# USE -> The PositionEmbedding class is responsible for creating positional encodings for the input sequences in a transformer model.
class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len,d_model)
        position_encoding = torch.zeros(seq_len, d_model)
        # create a vectpr of shape seq_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len,1)
        # create a vector of shape d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine on even indices
        position_encoding[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * 100000 ** (2i/d_model))
        # apply cosine on odd indices
        position_encoding[:, 1::2] = torch.cos(
            position * div_term
        )  # cos(position * 100000 ** (2i/d_model))
        # add the batch dimension to the position encoding
        position_encoding = position_encoding.unsqueeze(0)  # (1,seq_len,d_model)
        # Register the position ecoding as a buffer
        self.register_buffer("position_encoding", position_encoding)

    def forward(self, x):
        """Generates positional encodings for input sequences in a transformer model.

        The positional encodings are added to the input sequences using element-wise addition. The
        positional encodings are created based on the position and the dimension of the input
        sequences. The positional encodings are created using sine and cosine functions. The
        positional encodings are registered as buffer tensors in the module.
        """
        x = x + (self.position_encoding[:, : x.shape[1], :]).requires_grad(
            False
        )  # (batch,seq_len,d_model)
        return self.dropout(x)


# Residual Layer
# USed while making encoders and decoders
# The ResidualConnection class implements the residual connection mechanism in a neural network. It adds the input tensor to the output of a sublayer, after applying dropout and layer normalization.
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """Applies the residual connection mechanism to the input tensor.

        Adds the input tensor to the output of a sublayer after applying dropout and layer
        normalization.
        """
        return x + self.dropout(sublayer(self.norm(x)))


# MultiheadAttention block
# Takinng care of key value and query
# The MultiHeadAttentionBlock class implements the multi-head attention mechanism used in the Transformer model. It performs self-attention on the input sequence and combines the results from multiple attention heads.
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "0 is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # Performs the attention calculation between the query, key, and value tensors.
        d_k = query.shape[-1]
        # Formulas from the paper
        # (batch_size,h,seq_len,d_k) --> (batch,h,seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -> inf) to the position where mask == 0
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch_size, seq_len,seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch,h,seq_len,seq_len) ---> (batch,h,seq_len,d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """Applies multi-head self-attention to the input sequence.

        Divides the input into multiple heads and performs attention on each head separately.
        Combines the results from all heads to produce the final output.
        """

        query = self.w_q(q)  # (batch,seq_len,d_model) --> (batch,seq_len,d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch,seq_len,d_model) --> (batch_size,seq_len,h,d_k) --> (batch,h,seq_len,d_k)
        query = query.view(query.shape[0], query.shape[1].self.h.self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Combine all the heads together
        # (batch,h,seq_len,d_k) --> (batch,seq_len,h,d_k) -> (batch,seq_len,d_k)
        x = (
            x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        )  # (batch,seq_len,d_model) --> (batch,seq_len,d_model)

        # Multiply by Wo
        # (batch,seq_len,d_model) --> (batch,seq_len,d_model)
        return self.w_o(x)


# The EncoderBlock class is a module that represents one block of the encoder in a transformer model. It consists of a self-attention block and a feed-forward block, with residual connections and layer normalization applied between them.
class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attnetion_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attnetion_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """Apply self-attention mechanism to the input tensor using the self-attention block.

        Apply feed-forward transformation to the output of the self-attention block using the feed-
        forward block. Add residual connections and layer normalization between the self-attention
        block and the feed-forward block.
        """
        x = self.residual_connections[0](x, lambda x: self.self.attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


# The Encoder class is a module that consists of multiple layers and a layer normalization operation. It is used to encode the input tensor by passing it through each layer and applying layer normalization.
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        """Passes the input tensor through each layer in the layers list.

        Applies layer normalization to the output tensor.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# The DecoderBlock class is a module that represents a single block in the decoder of a transformer model. It consists of three main components: self-attention, cross-attention, and feed-forward layers. The class applies residual connections and layer normalization to the output of each component.
class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = self.cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """Apply self-attention to the input tensor x using the self_attention_block.

        Apply cross-attention to the input tensor x and the encoder output tensor encoder_output
        using the cross_attention_block. Apply the feed-forward block to the output of the cross-
        attention layer. Apply residual connections and layer normalization to the output of each
        component.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


# The Decoder class is a component of the Transformer model that performs the decoding process. It consists of multiple layers of decoder blocks, each containing self-attention and feed-forward sub-layers. The class applies layer normalization to the output of each decoder block.
class Decoder(nn.module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """The main functionality of the Decoder class is to perform the decoding process in the
        Transformer model.

        It applies multiple layers of decoder blocks to the input tensor. It applies layer
        normalization to the output of each decoder blo
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# The ProjectionLayer class is a module in the Transformer model that performs a linear projection followed by a log softmax activation function.
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """The ProjectionLayer class takes as input the dimension of the input tensor (d_model) and
        the size of the vocabulary (vocab_size).

        It performs a linear projection of the input tensor using a fully connected layer
        (nn.Linear) with d_model input features and vocab_size output features. The output of the
        linear projection is passed through a log softmax activation function (torch.log_softmax)
        along the last dimension (dim=-1). The resulting tensor is the log probabilities of the
        classes in the vocabulary.
        """
        return torch.log_softmax(self.proj(x), dim=-1)


# The Transformer class is the main class that implements the Transformer model in PyTorch. It consists of an encoder, a decoder, input embedding layers, positional encoding layers, and a projection layer. The class provides methods for encoding and decoding input sequences, as well as projecting the output of the decoder.
class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        projection_layer: ProjectionLayer,
        src_pos,
        tgt_pos,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    """
    Encode the source sequence using the encoder
    Decode the target sequence using the decoder and the encoded source sequence
    Project the output of the decoder to obtain the final output
    """

    def encode(self, src, src_mask):
        # Encodes the source sequence by passing it through the input embedding layer, positional encoding layer, and the encoder. Returns the encoded source sequence.
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> None:
        # Decodes the target sequence by passing it through the input embedding layer, positional encoding layer, and the decoder. Takes the encoded source sequence, source mask, target sequence, and target mask as inputs. Returns the decoded target sequence.
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, tgt_mask)

    def project(self, x):
        # Projects the input tensor x using the projection layer. Returns the projected output.
        return self.projection_layer(x)


# The build_transformer function is responsible for constructing a Transformer model by creating the necessary components such as embedding layers, positional encoding layers, encoder and decoder blocks, and the projection layer. It also initializes the parameters of the model.
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """Create embedding layers for the source and target inputs.

    Create positional encoding layers for the source and target sequences. Create a list of encoder
    blocks, each consisting of a self-attention block and a feed-forward block. Create a list of
    decoder blocks, each consisting of a self-attention block, a cross-attention block, and a feed-
    forward block. Create an encoder using the list of encoder blocks. Create a decoder using the
    list of decoder blocks. Create a projection layer for the output. Create a Transformer model
    using the encoder, decoder, embedding layers, positional encoding layers, and projection layer.
    Initialize the parameters of the model using Xavier uniform initialization. Return the
    Transformer model.
    """
    # Create a embedding layer
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layer
    src_pos = PositionEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionEmbedding(d_model, tgt_seq_len, dropout)

    # create the decoder block
    encoder_blocks = []
    for _ in range(N):  # Total 6 encoder blocks
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create a decoder blocks
    decoder_blocks = []
    for _ in range(N):  # Total 6 decoder blocks
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block.dropout,
        )
        decoder_blocks.append(decoder_block)

    # create a encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return (
        transformer  # The function returns a Transformer model with the specified configuration.
    )
