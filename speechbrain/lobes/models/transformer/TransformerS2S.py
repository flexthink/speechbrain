from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
    get_key_padding_mask,
)

import torch
from torch import nn


class TransformerS2S(TransformerInterface):
    """A Transformer for a generic sequence-to-sequence task where one sequence
    of tokens is converted to another (such as vanilla grapheme-to-phoneme, text-to-speech
    with tokenized audio, etc)

    Arguments
    ----------
    src_vocab: int
        the source vocabulary size
    tgt_vocab: int
        the target vocabulary size
    src_embedding_dim: int
        the embedding dimension for source tokens
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers: int, optional
        The number of encoder layers in1ì the encoder.
    num_decoder_layers: int, optional
        The number of decoder layers in the decoder.
    dim_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    custom_src_module: torch.nn.Module, optional
        Module that processes the src features to expected feature dim.
    custom_tgt_module: torch.nn.Module, optional
        Module that processes the src features to expected feature dim.
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        Choose between Conformer and Transformer for the encoder. The decoder is fixed to be a Transformer.
    conformer_activation: torch.nn.Module, optional
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    pad_idx: int
        the padding index (for masks)
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size=15,
        bias=True,
        encoder_module="transformer",
        attention_type="regularMHA",
        max_length=2500,
        causal=False,
        pad_idx=0,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
        )
        self.pad_idx = pad_idx
        self._reset_params()

    def forward(
        self,
        src,
        tgt,
        src_length,
        tgt_length,
    ):
        """Computes the forward pass

        Arguments
        ---------
        src : torch.Tensor
            the source token sequence
        tgt : torch.Tensor
            the target token sequence
        src_length : torch.Tensor
            the source relative lengths
        tgt_length : torch.Tensor
            the target relative lengths

        Returns
        -------
        encoder_out : torch.Tensor
            encoder outputs

        decoder_out : torch.Tensor
            decoder outputs
        """

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, src_length, pad_idx=self.pad_idx)

        pos_embs_encoder = None
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None

        encoder_out, encoder_self_attns = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            src = src + self.positional_encoding_decoder(src)
            pos_embs_encoder = None
            pos_embs_target = None
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        decoder_out, decoder_self_attns, decoder_multihead_attns = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return encoder_out, decoder_out, encoder_self_attns, decoder_self_attns, decoder_multihead_attns

    def _reset_params(self):
        """Resets the parameters of the model"""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def make_masks(self, src, tgt, src_length=None, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        src_length : tensor
            The sequence length to the encoder(required)
        pad_idx : int
            The index for <pad> token (default=0).


        Returns
        -------
        src_key_padding_mask: torch.Tensor
            the source key padding mask
        tgt_key_padding_mask: torch.Tensor
            the target key padding masks
        src_mask: torch.Tensor
            the source mask
        tgt_mask: torch.Tensor
            the target mask
        """
        if src_length is not None:
            abs_len = torch.round(src_length * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    def decode(self, tgt, encoder_out, enc_lens):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.

        Returns
        -------
        prediction: torch.Tensor
            the predicted sequence
        attention: torch.Tensor
            the attention matrix corresponding to the last attention head
            (useful for plotting attention)
        """
        tgt_mask = get_lookahead_mask(tgt)
        if self.attention_type == "RelPosMHAXL":
            # we use fixed positional encodings in the decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            encoder_out = encoder_out + self.positional_encoding_decoder(
                encoder_out
            )
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            pos_embs_tgt=None,
            pos_embs_src=None,
        )
        attention = multihead_attns[-1]
        return prediction, attention