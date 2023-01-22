"""
MadMixture shared latent space multimodal audio model

Authors
 * Artem Ploujnikov 2022-2023
"""
import torch
from torch import nn, Tensor
from typing import Dict
from speechbrain.nnet.attention import MultiheadAttention
from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.callchains import arg_exists
from math import floor, ceil
from enum import Enum

class MadMixture(nn.Module):
    """The MAdmixture (Modality Admixture) generic multimodal
    model framework
    
    Arguments
    ---------
    modalities: list[Modality]|dict
        a list of modality specifications with encoder/decoder models

    anchor_name: str
        the name of the "anchor" modality, i.e. the one with which
        all other modalities are aligned

    latent_size: int
        the size of the feature dimension of the latent space
    """
    def __init__(self, modalities, anchor_name, latent_size=32):
        super().__init__()
        if isinstance(modalities, dict):
            self.modalities = {
                name: Modality(name=name, **modality_kwargs)
                for name, modality_kwargs in modalities.items()
            }
        else:
            self.modalities = {
                modality.name: modality
                for modality in modalities            
            }
        self.anchor_name = anchor_name
        self.anchor = self.modalities[anchor_name]
        self.latent_size = latent_size
        self.aligned_modalities = {
            key
            for key, modality in self.modalities.items()
            if modality.aligner
        }
        modalities = set(self.modalities.keys())
        self.unaligned_modalities = modalities - self.aligned_modalities

    def forward(self, inputs, lengths=None, context=None):
        """Runs the forward pass (encodes inputs)

        Arguments
        ---------
        inputs: dict
            A string -> tensor dictionary representing the model inputs

        lengths: dict
            A string -> tensor dictionary representing relative lengths
            for variable-length inputs, where applicable

        context: dict
            arbitrary model-specific context information, as a
            str -> tensor dictionary. This can be used to pass items like
            ground truths for teacher forcing, alignment data, etc
            
        Returns
        -------
        outputs: dict
            A string->tensor dictionary with encoder outputs """
        
        self.encode(inputs, lengths, context)

    def encode(self, inputs, lengths=None, context=None):
        """Encodes the inputs from each modality into the latent space
        with the corresponding encoder
        
        Arguments
        ---------
        inputs: dict
            A string -> tensor dictionary representing the model inputs

        lengths: dict
            A string -> tensor dictionary representing relative lengths
            for variable-length inputs, where applicable

        context: dict
            arbitrary model-specific context information, as a
            str -> tensor dictionary. This can be used to pass items like
            ground truths for teacher forcing, alignment data, etc

            
        Returns
        -------
        outputs: dict
            A string->tensor dictionary with encoder outputs """
        
        if lengths is None:
            lengths = full_lengths(inputs)

        return {
            key: self.modalities[key].encode(
                input,
                lengths=lengths.get(key),
                context=context
            )
            for key, input in inputs.items()
        }
    
    def latent(self, inputs, lengths=None, context=None):
        """Computes the latent representations for each modality
        by first running the encoders and then running the aligners,
        if provided

        Arguments
        ---------
        inputs: dict
            A string -> tensor dictionary representing the model inputs

        lengths: dict
            A string -> tensor dictionary representing relative lengths
            for variable-length inputs, where applicable

        context: dict
            arbitrary model-specific context information, as a
            str -> tensor dictionary. This can be used to pass items like
            ground truths for teacher forcing, alignment data, etc


        Returns
        -------
        latents: dict
            a key -> tensor dictionary of aligned latent representations
        alignments: dict
            a key -> tensor dictionary of alignment matrices, only for
            modalities that support alignment
        enc_out: dict
            raw encoder ouputs for each modality        
        """
        if lengths is None:
            lengths = full_lengths(inputs)
        enc_out = self.encode(inputs, lengths, context)
        aligner_out = {
            key: self.alignment(key, enc_out, lengths)
            for key in inputs
            if key in self.aligned_modalities
        }
        latents = {
            key: latent
            for key, (latent, _) in aligner_out.items()
        } | {
            key: enc_out[key]
            for key in self.unaligned_modalities
        }
        alignments = {
            key: alignment
            for key, (alignment, _) in aligner_out.items()
        }
        return latents, alignments, enc_out

    def alignment(self, key, enc_out, lengths):
        """Computes the alignment for the specified modality

        Aguments
        --------
        key: str
            the modality key
        enc_out
            raw encoder outputs for all modalities
        length: torch.Tensor
            relative lengths
            

        Returns
        -------
        result: torch.Tensor
            the aligned tensor
        
        alignment: torch.Tensor
            the raw alignment matrix
        """
        if key in self.aligned_modalities:
            result = self.modalities[key].aligner(key, enc_out, lengths)
        else:
            result = None, None
        return result


    def decode_single(self, latent, lengths, context=None):
        """Decodes a single latent-space representation to all modalities. This
        can be useful during inference (e.g. to reconstruct one modality from another
        where applicable
        
        Arguments
        ---------
        latent: torch.Tensor
            a single latent space batch
        
        length: torch.Tensor
            the lengths of individual items"""
        return {
            key: modality.decode(latent, lengths=lengths, context=context)
            for key, modality in self.modalities.items()
        }

    def decode_multiple(self, latent, lengths, context=None):
        """Decodes multiple latent-space representations using the decoder for
        each corresponding modality. This can be useful during training
        
        Arguments
        ---------
        latent: dict
            A str -> tensor dictionary representing a latent space representation
            for multiple modalities
        
        Returns
        -------
        rec: dict
            A str -> tensor dictionary representing reconstruction
        
        """
        return {
            key: self.modalities[key].decode(
                modality_latent,
                lengths=lengths.get(key) if lengths is not None else None,
                context=context)
            for key, modality_latent in latent.items()
        }
    
    def train_step(self, inputs, lengths=None, context=None):
        """A convenience function for model training, 
        encoding inputs and then decoding them from
        latent representations

        Arguments
        ---------
        inputs: dict
            A string -> tensor dictionary representing the model inputs

        lengths: dict
            A string -> tensor dictionary representing relative lengths
            for variable-length inputs, where applicable

        context: dict
            arbitrary model-specific context information, as a
            str -> tensor dictionary. This can be used to pass items like
            ground truths for teacher forcing, alignment data, etc
        
        """
        latents, alignments, enc_out = self.latent(inputs, lengths, context)
        rec = self.decode_multiple(latents, lengths, context)
        return latents, alignments, enc_out, rec
    

class ModalityType(Enum):
    PRIMARY = "primary"
    AUXILIARY = "auxiliary"
    

def full_lengths(inputs):
    """Returns a full relative length tesnor for every modality
    
    Arguments
    ---------
    inputs: dict[str, torch.Tensor]
        a dictionary of multimodal inputs

    Returns
    -------
    lengths: torch.Tensor
        a dictionary of lengths corresponding to the
        inputs
    """
    return {
        key: torch.ones(input_value.size(0))
        for key, input_value in inputs.items()
    }


# NOTE: This is a series of hacks in order to allow for
# modules with different signatures to be passed without sacrificing
# jittability given the lack of kwargs support in TorchScript
class CallWrapper(nn.Module):
    """A superclass for call wrappers to wrap encoders and decoders.
    The default implementation drops everything except the data tensor (x)

    Arguments
    ---------
    module: torch.nn.Module 
        the module to be wrapped
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, lengths, context):
        """Invokes the module, discarding the lengths and context

        Arguments
        ---------
        x: torch.Tensor
            the data tensor (inputs or latents)

        length: torch.Tensor
            a length tensor

        context: dict
            arbitrary model-specific context information, as a
            str -> tensor dictionary. This can be used to pass items like
            ground truths for teacher forcing, alignment data, etc

        Returns
        -------
        result: torch.Tensor
            the encoder or decoder output
        """
        return self.module(x)


class LengthsCallWrapper(CallWrapper):
    """A wrapper for modules with a lengths argument"""
    def forward(self, x, lengths, context):
        """Invokes the module, discarding only the context

        Arguments
        ---------
        x: torch.Tensor
            the data tensor (inputs or latents)

        length: torch.Tensor
            a length tensor

        context: Dict[str, torch.Tensor]
            arbitrary model-specific context information

        Returns
        -------
        result: torch.Tensor
            the encoder or decoder output
        """
        return self.module(x, lengths)


class ContextCallWrapper(CallWrapper):
    """A wrapper for the full signature"""
    def forward(self, x, lengths, context):
        """Invokes the module, discarding the lengths and context

        Arguments
        ---------
        x: torch.Tensor
            the data tensor (inputs or latents)

        length: torch.Tensor
            a length tensor

        context: Dict[str, torch.Tensor]
            arbitrary model-specific context information

        Returns
        -------
        result: torch.Tensor
            the encoder or decoder output
        """
        return self.module(x, lengths, context)
    

def wrap_module(module):
    """Wraps the module with a call wrapper with the appropriate signature
    
    Arguments
    ---------
    torch.nn.Module
        the module to be wrapped, with one of the following signatures
        (x)
        (x, lengths)
        (x, lengths, context)
    """
    if arg_exists(module.forward, "context"):
        wrapper = ContextCallWrapper
    elif arg_exists(module.forward, "lengths"):
        wrapper = LengthsCallWrapper
    else:
        wrapper = CallWrapper

    return wrapper(module)


class Modality(nn.Module):
    """Represents a single modality within the MadMixture model
    
    Arguments
    ---------
    name: str
        the modality identifier
    
    encoder: nn.Module
        the encoder for the modality
    
    decoder: nn.Module
        the decoder for the modality

    aligner: nn.Module
        an optional aligner module, aligning the outputs
        to the latent representation

    """
    def __init__(self, name, encoder, decoder, aligner=None):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.aligner = aligner
        self.encoder = wrap_module(encoder)
        self.decoder = wrap_module(decoder)

    def forward(self, input, lengths=None):
        """Performs the encode operation (the default)
        
        Arguments
        ---------
        input: torch.Tensor
            an input tensor for the modality
        lengths: torch.Tensor
            relative lengths
        """
        return self.encode(input, lengths)
    
    def encode(self, input, lengths=None, context=None):
        # type: (Tensor, Tensor, Dict[str, Tensor]) -> Tensor
        return self.encoder(input, lengths, context)
    
    def decode(self, latent, lengths=None, context=None):
        # type: (Tensor, Tensor, Dict[str, Tensor]) -> Tensor
        return self.decoder(latent, lengths, context)


class Aligner(nn.Module):
    """A base class for auxiliary modules that align encoded outputs in
    a given modality to the latent space"""
    def forward(self, key, enc_out, length):
        raise NotImplementedError()



class AttentionalAligner(nn.Module):
    """An aligner that uses an attention mechanism to align the raw encoder
    outputs to the target latent space
    dim: int
        the output dimension
    in_dim: int
        the input dimension (defaults to the output dimension)
    nhead: int
        the number of attention heads
    dropout: float
        the dropout probability
    kernel_size: int
        the kernel size 
    scale: float
        the scaling factor
    """

    def __init__(self, dim, in_dim=None, nhead=1, dropout=0., scale=None, kernel_size=None):
        super().__init__()
        if in_dim is None:
            in_dim = dim
        self.in_dim = in_dim
        self.attn = MultiheadAttention(
            nhead=nhead,
            d_model=dim,
            dropout=dropout
        )
        if scale is None:
            scale = 1
        self.scale = scale
        if (scale - 1 < 0.01) and in_dim == dim:
            # No scaling
            self.scale = 1
            self.compressor = nn.Identity()
        elif scale > 1.:
            stride = ceil(scale)
            if kernel_size is None:
                kernel_size = stride
            padding = (kernel_size - stride) // 2
            self.compressor = ConvTranspose1d(
                in_channels=in_dim,
                out_channels=dim,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding
            )
        else:
            stride = floor(1 / scale)
            if kernel_size is None:
                kernel_size = scale
            self.compressor = Conv1d(
                in_channels=in_dim,
                out_channels=dim,
                stride=stride,
                kernel_size=kernel_size,
                padding="same"
            )
        
        # Rescale the feature dimension only using a convolutional layer, if necessary
        if in_dim != dim:
            self.feature_scale = Conv1d(
                in_channels=in_dim, out_channels=dim, padding="same", kernel_size=1)
        else:
            self.feature_scale = nn.Identity()

    def get_queries(self, enc_out):
        """Upsamples or downsamples raw encoder outputs
        before applying attention   
        
        """
        enc_out_scaled = self.compressor(enc_out)
        new_length = floor(enc_out.size(1) * self.scale)
        return enc_out_scaled[:, :new_length, :]

    def forward(self, key, enc_out, lengths):
        """Computes alignments using an attention module

        Arguments
        ---------
        key: str
            the modality key
        enc_out: torch.Tensor
            raw encoded states
        length: torch.Tensor
            relative lengths
        
        """
        mod_enc_out = enc_out[key]
        mod_length = lengths[key]
        queries = self.get_queries(mod_enc_out)
        max_len = queries.size(1)
        mask = length_to_mask(mod_length * max_len, max_len).unsqueeze(-1)
        masked_queries = queries * mask
        mod_enc_out_scaled = self.feature_scale(mod_enc_out) * mask
        output, alignment = self.attn(masked_queries, mod_enc_out_scaled, mod_enc_out_scaled)
        return output * mask, alignment
