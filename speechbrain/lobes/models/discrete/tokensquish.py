"""
A simple transformer-based autoencoder to compress long sequences
of tokens

Author
 * Artem Ploujnikov

"""
import torch
from collections import namedtuple
from torch import nn
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.dataio.dataio import length_to_mask, clean_padding_
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    PositionalEncoding,
)
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.embedding import MultiEmbedding


class TokenSquishModel(nn.Module):
    """A compression model for sequences of tokens. This model can be used to reduce
    the total sequence length for discrete audio representations

    Arguments
    ---------
    num_tokens : int, optional
        The number of tokrnds available
    tokens_per_step : int, optional
        The number of tokens per step
    embedding_dim : int, optional
        The embedding dimension
    d_model : int, optional
        The transformer model dimension
    nhead : int, optional
        The number of attention heads
    enc_num_layers : int, optional
        The number of encoder layers
    dec_num_layers: int, optional
        The number of decoder layers
    dropout : float, optional
        The dropout probability
    scale_factor : int, optional
        The scalling factor by which the length will be reduced
    activation : torch.nn.Module, optional
        The activation function to be used
    attention_type : str
        The type of transformer attention to be used
    interpolation_mode: str
        The interpolation mode to be used for token upsampling
    max_len : int
        The maximum sequence length
    """

    def __init__(
        self,
        num_tokens=1024,
        tokens_per_step=2,
        embedding_dim=256,
        d_model=512,
        d_ffn=2048,
        nhead=4,
        enc_num_layers=3,
        dec_num_layers=3,
        dropout=0.2,
        scale_factor=2,
        activation=nn.LeakyReLU,
        attention_type="RelPosMHAXL",
        interpolation_mode="linear",
        max_len=5000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale_factor = scale_factor
        self.num_tokens = num_tokens
        self.tokens_per_step = tokens_per_step
        self.d_model = d_model
        self.attention_type = attention_type
        self.interpolation_mode = interpolation_mode
        self.emb = MultiEmbedding(
            num_embeddings=num_tokens,
            embedding_dim=embedding_dim,
            num_heads=tokens_per_step,
        )
        if attention_type == "RelPosMHAXL":
            self.pos_emb = RelPosEncXL(emb_dim=d_model,)
            self.pos_emb_dec = PositionalEncoding(
                input_size=d_model, max_len=max_len
            )
        else:
            self.pos_emb = self.pos_emb_dec = PositionalEncoding(
                input_size=d_model, max_len=max_len
            )
        self.in_proj = Linear(
            input_size=embedding_dim * tokens_per_step, n_neurons=d_model
        )
        self.enc = TransformerEncoder(
            num_layers=enc_num_layers,
            nhead=nhead,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            normalize_before=True,
            attention_type=attention_type,
        )
        self.enc_out_norm = LayerNorm(input_size=d_model,)
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode=interpolation_mode,
        )
        self.dec = TransformerEncoder(
            num_layers=dec_num_layers,
            nhead=nhead,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            normalize_before=True,
            attention_type="regularMHA",
        )
        self.dec_out_norm = LayerNorm(input_size=d_model,)
        self.out_proj = Linear(
            input_size=d_model, n_neurons=num_tokens * tokens_per_step
        )

    def encode(self, tokens, length):
        """Encodes a token sequence to a latent representation

        Arguments
        ---------
        tokens : torch.Tensor
        `   A sequence of tokens (Batch x Length)
        length : torch.Tensor
            Relative sequence length

        Returns
        -------
        """
        src_key_padding_mask = self.make_mask(tokens, length)
        emb = self.emb(tokens)
        batch_size, seq_len, num_heads, emb_dim = emb.shape
        emb = emb.reshape(batch_size, seq_len, num_heads * emb_dim)
        enc_in = self.in_proj(emb)
        enc_in, pos_embs = self.get_pos_embs(enc_in)
        enc_out, self_attn = self.enc(
            enc_in,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )
        latents = self.enc_out_norm(enc_out)
        clean_padding_(latents, length)
        enc_attn = torch.stack(self_attn, dim=1)
        return TokenSquishEncodeOutput(latents=latents, enc_attn=enc_attn,)

    def get_pos_embs(self, src):
        """Applies positional embeddings to a tensor

        Arguments
        ---------
        src : torch.Tensor
            the original tensor

        Returns
        -------
        out : torch.Tensor
            The original tensor (for relative attention) or the original tensor
            plus positional embeddings (for absolute attention)
        pos_emb : torch.Tensor
            The positional embeddings to be passed into the model"""
        if self.attention_type == "RelPosMHAXL":
            out = src
            pos_embs = self.pos_emb(src)
        else:
            out = src + self.pos_emb(src)
            pos_embs = None
        return out, pos_embs

    def decode_latent_as_probs(self, latent, length):
        """Decodes latent representations as token probabilities

        Arguments
        ---------
        latent : torch.Tensor
            The latent representation (from the encoder)
        length : torch.Tensor
            Relative lengths

        Returns
        -------
        p_seq : torch.Tensor
            A sequence of token log-probabilities
        dec_attn : torch.Tensor
            Decoder attention alignments
        """
        batch_size, max_len, _, = latent.shape
        dec_in = self.upsample(latent.transpose(-1, -2)).transpose(-1, -2)
        src_key_padding_mask = self.make_mask(dec_in, length)
        dec_in = dec_in + self.pos_emb_dec(dec_in)
        dec_out, self_attn = self.dec(
            dec_in, src_key_padding_mask=src_key_padding_mask, pos_embs=None
        )
        dec_out = self.dec_out_norm(dec_out)
        p_seq = self.out_proj(dec_out)
        batch_size, seq_len, _ = p_seq.shape
        p_seq = p_seq.reshape(
            batch_size, seq_len, self.tokens_per_step, self.num_tokens
        )
        p_seq = torch.log_softmax(p_seq, -1)
        dec_attn = torch.stack(self_attn, dim=1)
        return TokenSquishDecodeOutput(p_seq, dec_attn)

    def make_mask(self, src, length):
        """Creates the source key padding mask

        Arguments
        ---------
        src : torch.Tensor
            The tensor to be passed to the Transformer model
        length : torch.Tensor
            Relative length

        Returns
        -------
        src_key_padding_mask : torch.Tensor
            The key padding mask
        """
        abs_len = torch.round(length * src.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()
        return src_key_padding_mask

    def forward(self, tokens, length):
        """Computes the forward pass

        Arguments
        ---------
        tokens : torch.Tensor
            A compressed sequence of tokens
        length : torch.Tensor
            Relative sequence lengths

        Returns
        -------
        p_seq : torch.Tensor
            The log-probabilities of recovered tokens
        enc_attn : torch.Tensor
            Encoder attention alignments
        dec_attn : torch.Tensor
            Decoder attention alignments
        latents : torch.Tensor
            The latent representation
        """
        enc_out = self.encode(tokens, length)
        p_seq, dec_attn = self.decode_latent_as_probs(enc_out.latents, length)
        return TokenSquishOutput(
            p_seq=p_seq,
            enc_attn=enc_out.enc_attn,
            dec_attn=dec_attn,
            latents=enc_out.latents,
        )

    def compress(self, tokens, length=None):
        """Compresses the token sequence

        Arguments
        ---------
        tokens : torch.Tensor
            A token sequence
        length : torch.Tensor, optional
            Relative lengths. They are not required because the implementation simply
            leaves every scale_factorth token, removing the others. It is provided in
            for future compatibility with other approaches"""

        return tokens[:, :: self.scale_factor]

    def decompress(self, tokens, length):
        if length is None:
            length = torch.ones(
                len(tokens), device=tokens.device, dtype=torch.float32
            )
        out = self(tokens, length)
        return out.p_seq.argmax(-1)


TokenSquishOutput = namedtuple(
    "TokenSquishOutput", ["p_seq", "enc_attn", "dec_attn", "latents"]
)

TokenSquishEncodeOutput = namedtuple(
    "TokenSquishEncodeOutput", ["latents", "enc_attn"]
)

TokenSquishDecodeOutput = namedtuple(
    "TokenSquishEncodeOutput", ["p_seq", "dec_attn"]
)
