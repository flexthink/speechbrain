"""A simplistic Text-to-Speech model operating on
discrete/tokenized audio representations

NOTE: This model does not use the standard Transformer interface
in order to make it usable as both as a full model and as a 
decoder-only model

Authors
* Artem Ploujnikov, 2023
"""

import torch
from torch import nn
from torch.nn import functional as F
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    TransformerDecoder,
    NormalizedEmbedding,
    PositionalEncoding,
    get_lookahead_mask,
)
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.linear import Linear
from speechbrain.dataio.dataio import length_to_mask
from collections import namedtuple
from tqdm.auto import tqdm

TokotronOutput = namedtuple(
    "TokotronOutput",
    [
        "out",
        "gate_out",
        "enc_self_attn",
        "dec_self_attn",
        "dec_attn",
        "alignments",
    ]
)

TokotronDecoderOutput = namedtuple(
    "TokotronDecoderOutput",
    [
        "out",
        "gate_out",
        "dec_self_attn",
        "dec_attn",
        "alignments",
    ]
)

TokotronDecoderInfernceOutput = namedtuple(
    "TokotronDecoderInferenceOutput",
    [
        "audio_tokens",
        "length",
        "dec_self_attn",
        "dec_attn",
        "alignments",
    ]
)

TokotronInfernceOutput = namedtuple(
    "TokotronInferenceOutput",
    [
        "audio_tokens",
        "length",
        "wav",
        "wav_length",
        "enc_self_attn",
        "dec_self_attn",
        "dec_attn",
        "alignments",
    ]
)


class TokotronDecoder(nn.Module):
    """The Tokotron decoder - can be used in a standalone model or as
    a component of a larger model
    
    Arguments
    ---------
    num_tokens : int
        the number of tokens
    tokens_per_step : int
        the number of tokens to be output, per transformer time step
    d_model : int
        The number of expected features in the encoder/decoder inputs (default=512).
    d_ffn : int, optional
        The dimension of the feedforward network model hidden layer.        
    nhead : int
        The number of heads in the multi-head attention models (default=8).
    audio_emb : torch.nn.Module
        The audio embedding to be used
    activation : torch.nn.Module
        The activation function to be used
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference

    """
    def __init__(
        self,
        num_tokens=1024,
        tokens_per_step=2,
        d_model=512,
        d_ffn=2048,
        nhead=4,
        attention_type="regularMHA",
        num_decoder_layers=6,
        dropout=0.2,
        target_dropout=None,
        bos_idx=0,
        audio_emb=None,
        activation=nn.LeakyReLU,
        max_decoder_steps=1000,
        show_inference_progress=True,
        gate_threshold=0.5,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens_per_step = tokens_per_step
        self.bos_idx = 0
        self.dec = TransformerDecoder(
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            num_layers=num_decoder_layers,
            activation=activation,
            dropout=dropout,
        )
        self.tgt_in_proj = Linear(
            input_size=d_model * tokens_per_step,
            n_neurons=d_model,
        )
        self.out_proj = Linear(
            input_size=d_model,
            n_neurons=num_tokens * tokens_per_step,
        )
        self.gate = Linear(
            input_size=d_model,
            n_neurons=1
        )
        if audio_emb is None:
            audio_emb = NormalizedEmbedding(
                d_model=d_model,
                vocab=num_tokens
            )
        self.positional_encoding = PositionalEncoding(
            d_model, max_decoder_steps
        )
        if target_dropout is None:
            target_dropout = dropout
        self.target_dropout = dropout
        self.audio_emb = audio_emb
        self.bos_idx = bos_idx
        self.max_decoder_steps = max_decoder_steps
        self.show_inference_progress = show_inference_progress
        self.attention_type = attention_type
        self.gate_threshold = gate_threshold

    def forward(
        self,
        enc_out,
        tgt,
        src_length=None,
        src_key_padding_mask=None,
        tgt_length=None,
        tgt_key_padding_mask=None,
        pos_embs_src=None,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        src : torch.Tensor
            Raw encoder outputs
        tgt : torch.Tensor
            Targets (audio tokens)
        src_length : torch.Tensor
            The relative lengths of the source sequence
        tgt_length : torch.Tensor
            Target lengths
        """
        if src_length is not None and src_key_padding_mask is None:
            src_max_len = enc_out.size(1)
            src_key_padding_mask = length_to_mask(
                src_length * src_max_len,
                src_max_len
            ).logical_not()
        if tgt_length is not None and tgt_key_padding_mask is None:
            tgt_max_len = tgt.size(1)
            tgt_key_padding_mask = length_to_mask(
                tgt_length * tgt_max_len,
                tgt_max_len
            ).logical_not()

        audio_emb = self.audio_emb(tgt)

        batch_size, audio_max_len, heads, audio_dim = audio_emb.shape
        audio_emb_combined = audio_emb.reshape(
            batch_size,
            audio_max_len,
            heads * audio_dim
        )
        tgt = self.tgt_in_proj(audio_emb_combined)
        tgt = F.dropout(
            tgt,
            self.target_dropout,
            training=self.training
        )

        tgt_mask = get_lookahead_mask(tgt)
        if self.attention_type == "RelPosMHAXL":
            pos_embs_tgt = self.positional_encoding(tgt)
        else:
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_tgt = None
        (
            dec_out,
            dec_self_attn,
            dec_attn,
        ) = self.dec(
            tgt=tgt,
            memory=enc_out,
            memory_mask=None,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_tgt,
            pos_embs_src=pos_embs_src,
        )

        lin_out = self.out_proj(dec_out)
        batch_size, text_max_len, _ = lin_out.shape
        lin_out_heads = lin_out.reshape(
            batch_size,
            text_max_len,
            self.tokens_per_step,
            self.num_tokens,
        )
        gate_out = self.gate(dec_out).squeeze(-1)
        return TokotronDecoderOutput(
            lin_out_heads,
            gate_out,
            dec_self_attn,
            dec_attn,
            get_alignments(dec_attn),
        )

    def get_bos(self, batch_size, device="cpu"):
        """Constructs a beginning-of-sequence (BOS) sequence for
        autoregressive inference

        Arguments
        ---------
        batch_size : int
            The size of the batch dimension
        device : str|torch.Device
            The device identifier

        Returns
        -------
        seq: torch.Tensor
            the target sequence"""
        return torch.ones(
            batch_size,
            1,
            self.tokens_per_step,
            device=device
        ) * self.bos_idx

    def infer(self, enc_out, length):
        """Performs autoregressive inference

        Arguments
        ---------
        enc_out : torch.Tensor
            Raw encoder outputs

        length : torch.Tensor
            Relative lengths

        Returns
        -------
        audio_tokens : torch.Tensor
            A (Batch x Length x Tokens) tensor of audio tokens
        length : torch.Tensor
            Inferred relative lengths
        dec_self_attn : torch.Tensor
            Decoder self-attentions
        dec_attn : torch.Tensor
            Decoder multihead attentions (or equivalent)
        """
        with torch.no_grad():
            batch_size = enc_out.size(0)
            bos = self.get_bos(batch_size, device=enc_out.device)
            audio_tokens = bos
            audio_tokens_length = torch.ones(batch_size, device=enc_out.device)

            steps_range = range(self.max_decoder_steps)
            if self.show_inference_progress:
                steps_range = tqdm(steps_range, desc="Inference")
            for _ in steps_range:
                step_out = self.forward(
                    enc_out=enc_out,
                    src_length=length,
                    tgt=audio_tokens,
                    tgt_length=audio_tokens_length,
                )
                audio_tokens_out = step_out.out.argmax(-1)
                audio_tokens = torch.cat(
                    [bos, audio_tokens_out],
                    dim=1
                )
                seq_done = F.sigmoid(step_out.gate_out) > self.gate_threshold
                done = seq_done.any(dim=-1).all()
                if done.item():
                    break

            length_abs = seq_done.int().argmax(dim=-1) + 1
            length_abs[length_abs == 1] = seq_done.size(1)
            length = length_abs.float() / seq_done.size(1)

        return TokotronDecoderInfernceOutput(
            audio_tokens=audio_tokens_out,
            length=length,
            dec_self_attn=step_out.dec_self_attn,
            dec_attn=step_out.dec_attn,
            alignments=step_out.alignments,
        )


class TokotronModel(nn.Module):
    """An end-to-end Tokotron model receiving characters or phonemes
    as inputs and outputting audio tokens

    Arguments
    ---------
    input_num_tokens : int
        The number of input characters or phonemes available
    audio_num_tokens : int
        The number of audio tokens
    audio_tokens_per_step : int
        The number of output audio tokens per tranformer step.
        When using Vocodec, this corresponds to the number of 
        quantizers in the model used
    d_model : int
        The number of expected features in the encoder/decoder inputs (default=512).
    d_ffn : int, optional
        The dimension of the feedforward network model hidden layer.        
    nhead : int
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of encoder layers in1ì the encoder.
    num_decoder_layers : int, optional
        The number of decoder layers in the decoder.
    dropout : int, optional
        The dropout value.
    target_dropout : int, optional
        The dropout probability for targets
    activation : torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    bos_idx : int
        the Beginning-of-Sequence index

    """
    def __init__(
        self,
        input_num_tokens,
        audio_num_tokens=1024,
        audio_tokens_per_step=2,
        d_model=512,
        d_ffn=2048,
        nhead=4,
        attention_type="regularMHA",
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.2,
        target_dropout=0.2,
        activation=nn.LeakyReLU,
        bos_idx=0,
        vocoder=None,
        max_input_length=1000,
    ):
        super().__init__()
        self.in_emb = Embedding(
            num_embeddings=input_num_tokens,
            embedding_dim=d_model,
        )
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = TokotronDecoder(
            num_tokens=audio_num_tokens,
            tokens_per_step=audio_tokens_per_step,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            num_decoder_layers=num_decoder_layers,
            bos_idx=bos_idx,
            activation=activation,
            dropout=dropout,
            target_dropout=target_dropout,
        )
        self.bos_idx = bos_idx
        self.vocoder = vocoder
        self.attention_type = attention_type
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
        else:
            self.positional_encoding = PositionalEncoding(
                d_model, max_input_length
            )

    def forward(
        self,
        input_tokens,
        input_length,
        audio_tokens,
        audio_length,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths
        audio_tokens : torch.Tensor
            a (Batch x Length) tensor of output audio tokens (e.g. encodec)
        audio_length : torch.Tensor
            a 1-D tensor of relative output lengths"""

        src, src_key_padding_mask, pos_embs_encoder = self.process_inputs(
            input_tokens, input_length
        )

        enc_out, enc_self_attn = self.encoder(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        tgt_mask = get_lookahead_mask(audio_tokens)
        dec_out = self.decoder(
            enc_out=enc_out,
            tgt=audio_tokens,
            tgt_mask=tgt_mask,
            tgt_length=audio_length,
            src_length=input_length,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs_src=pos_embs_encoder,
        )
        return TokotronOutput(
            out=dec_out.out,
            gate_out=dec_out.gate_out,
            enc_self_attn=enc_self_attn,
            dec_self_attn=dec_out.dec_self_attn,
            dec_attn=dec_out.dec_attn,
            alignments=dec_out.alignments,
        )
    
    def process_inputs(self, input_tokens, input_length):
        """Computes embeddings, the padding mask and encoder
        positional embeddings

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths

        Returns
        -------
        src : torch.Tensor
            input embeddings
        src_key_padding_mask : torch.Trnsor
            the key padding mask for inputs
        pos_emb_encoder : torch.Tensor
            encoder positional embeddings
        """
        in_emb = self.in_emb(input_tokens)
        pos_embs_encoder = None
        if self.attention_type == "RelPosMHAXL":
            src = in_emb
            pos_embs_encoder = self.positional_encoding(in_emb)
        else:
            src = in_emb + self.positional_encoding(in_emb)  # add the encodings here
            pos_embs_encoder = None

        input_max_len = input_tokens.size(1)
        src_key_padding_mask = length_to_mask(
            input_length * input_max_len,
            input_max_len,
        ).logical_not()
        return src, src_key_padding_mask, pos_embs_encoder

    def infer(self, input_tokens, input_length):
        """Performs end-to-end inference
        
        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths

        Returns
        -------
        audio_tokens : torch.Tensor
            A (Batch x Length x Tokens) tensor of audio tokens
        length : torch.Tensor
            Inferred relative lengths
        wav : torch.Tensor
            Synthesized waveforms, if a vocoder is provided
        wav_length : torch.Tensor
            Waveform lengths
        enc_self_attn : torch.Tensor
            Encoder self-attentions
        dec_self_attn : torch.Tensor
            Decoder self-attentions
        dec_attn : torch.Tensor
            Decoder multihead attentions (or equivalent)

        """
        src, src_key_padding_mask, pos_embs_encoder = self.process_inputs(
            input_tokens, input_length
        )
        enc_out, enc_self_attn = self.encoder(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        dec_out = self.decoder.infer(enc_out, input_length)
        wav, wav_length = None, None
        if self.vocoder is not None:
            wav, wav_length = self.vocoder(
                dec_out.audio_tokens,
                input_length
            )
        return TokotronInfernceOutput(
            audio_tokens=dec_out.audio_tokens,
            length=dec_out.audio_tokens,
            wav=wav,
            wav_length=wav_length,
            enc_self_attn=enc_self_attn,
            dec_self_attn=dec_out.dec_self_attn,
            dec_attn=dec_out.dec_attn,
            alignments=dec_out.alignments,
        )


def get_gate_targets(lengths, out_len):
    """Computes gate tarets and weights for each position

    Arguments
    ---------
    lengths : torch.Tensor
        Relative lengths
    out_len: int
        The maximum output length

    Returns
    -------
    tagrets : torch.Tensor
        Targets for gate outputs - EOS positions are marked as 1,
        non-EOS positions are marked at 0
    weights : torch.Tensor
        Weights by which individual position losses will be multiplied
    """
    pos = torch.arange(out_len, device=lengths.device)[None, :]
    gate_targets = pos >= (lengths * out_len)[:, None]
    gate_weights = torch.where(
        gate_targets,
        .5 / (1. - lengths)[:, None],
        .5 / lengths[:, None],
    )
    return gate_targets.float(), gate_weights


def get_alignments(attn):
    """Aggregates alignments from multiple layers and heads
    
    Arguments
    ---------
    attn: list
        raw attentions returned from a Transformer

    Results
    -------
    alignments: torch.Tensor
        The resulting alignments
    """
    return torch.cat(
        [item.unsqueeze(-1) for item in attn],
        dim=-1
    ).mean(dim=-1)
