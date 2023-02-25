"""
MadMixture shared latent space multimodal audio model

Authors
 * Artem Ploujnikov 2022-2023
"""
import torch
from collections import namedtuple
from torch import nn, Tensor
from typing import Dict, Tuple
from speechbrain.nnet.attention import MultiheadAttention
from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.callchains import arg_exists
from speechbrain.utils.data_utils import adjust_dim, concat_padded_features
from speechbrain.lobes.models.transformer.Transformer import (
    PositionalEncoding)
from math import floor, ceil
from enum import Enum

# TODO: The current handling of decoder lengths does not
# make a lot of sense - one would need to pass and predict
# anchor lengths for each modality

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
    
    modality_enabled: dict
        a dictionary with on-off switch setting which modalities will be enabled
        or disabled. This is a convenience parameter to be able to enable/disable
        modalities using hparams switches
    
    """
    def __init__(self, modalities, length_predictor, anchor_name=None, latent_size=32, modality_enabled=None):
        super().__init__()
        self.modality_enabled = modality_enabled
        if isinstance(modalities, dict):
            modalities_dict = {
                name: Modality(name=name, **modality_kwargs)
                for name, modality_kwargs in modalities.items()
                if self.is_enabled(name)
            }
        else:
            modalities_dict = {
                modality.name: modality
                for modality in modalities
                if self.is_enabled(modality.name)
            }
        self.modalities = nn.ModuleDict(modalities_dict)
        if not anchor_name:
            anchor_name = next(iter(modalities_dict.keys()))
        self.anchor_name = anchor_name
        self.anchor = self.modalities[anchor_name]
        self.latent_size = latent_size
        self.aligned_modalities = {
            key
            for key, modality in self.modalities.items()
            if modality.aligner
        }
        self.primary_modalities = {
            key            
            for key, modality in self.modalities.items()
            if modality.mod_type == ModalityType.PRIMARY
        }
        self.possible_transfers = [
            (src, tgt)
            for src in self.primary_modalities
            for tgt in self.primary_modalities
        ]
        self.cross_transfers = [
            (src, tgt)
            for src, tgt in self.possible_transfers
            if src != tgt
        ]
        modalities = set(self.modalities.keys())
        self.unaligned_modalities = modalities - self.aligned_modalities
        self.length_predictor = length_predictor
        self.eos_mark = EndOfSequenceMarker(feature_size=latent_size)

    def is_enabled(self, key):
        return self.modality_enabled.get(key, False) if self.modality_enabled is not None else True


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
        lengths_latent: dict
            post-alignment lengths     
        lengths_input: dict
            absolute lengths of inputs in the alignments
        """
        if lengths is None:
            lengths = full_lengths(inputs)
        enc_out = self.encode(inputs, lengths, context)
        latents, alignments, lengths_latent, lengths_input = {}, {}, {}, {}
        for key in inputs:
            if key in self.aligned_modalities:
                latent, alignment, length_latent, length_input = self.alignment(key, enc_out, lengths)
                latents[key] = latent
                alignments[key] = alignment
                lengths_latent[key] = length_latent
                lengths_input[key] = length_input
            else:
                latents[key] = enc_out[key]
                lengths_latent[key] = lengths[key]
                lengths_input[key] = lengths[key]
        return latents, alignments, enc_out, lengths_latent, lengths_input
    
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
        latent: torch.Tensor
            the aligned tensor
        
        alignment: torch.Tensor
            the raw alignment matrix

        lengths_latent: torch.Tensor
            post-alignment lengths
        """
        if key in self.aligned_modalities:
            if self.anchor_name in lengths:
                anchor_name = self.anchor_name
                anchor_scale = self.modalities[anchor_name].aligner.scale
            else:
                anchor_name, anchor_scale = None, None
            result = self.modalities[key].aligner(
                key, enc_out, lengths, anchor_name, anchor_scale, self.eos_mark)
        else:
            result = None, None, None
        return result


    def decode_single(self, latent, lengths, tgt=None, context=None):
        """Decodes a single latent-space representation to all modalities. This
        can be useful during inference (e.g. to reconstruct one modality from another
        where applicable
        
        Arguments
        ---------
        latent: torch.Tensor
            a single latent space batch

        target: list
            the target modalities    
        
        length: torch.Tensor
            the lengths of individual items

        Returns
        -------
        rec: dict
            the reconstruction from the latent space,
            from each modality
        context: dict
            the context collected from each encoder    
        """
        
        if tgt is None:
            targets = self.modalities.keys()
        elif isinstance(tgt, str):
            targets = [tgt]
        else:
            targets = tgt

        rec_with_context = {
            key: self.modalities[key].decode(latent, lengths=lengths, context=context)
            for key in targets
        }
        return self._extract_context(rec_with_context)

    def _extract_context(self, rec_with_context):
        """Extracts context information from a single
        dictionary with both reconstructions and context
        for each modality into two distinct dictionaries
        
        Arguments
        ---------
        rec_with_context: dict
            a str -> (rec, context) dictionary for all modalities
            
        Returns
        -------
        rec: dict
            a dictionary of reconstructions for each modality
        context: dict
            context entries from all modalities
        """
        rec, context = {}, {}
        for key, (mod_rec, mod_context) in rec_with_context.items():
            rec[key] = mod_rec
            prefixed_mod_context = {
                f"{key}_{context_key}": value
                for context_key, value in mod_context.items()
            }
            context.update(prefixed_mod_context)
        return rec, context
    
    def _extract_context_cross(self, rec_with_context):
        """Extracts context information from a single
        dictionary with both reconstructions and context
        for each modality into two distinct dictionaries
        
        Arguments
        ---------
        rec_with_context: dict
            a str -> (rec, context) dictionary for all modalities
            
        Returns
        -------
        rec: dict
            a dictionary of reconstructions for each modality
        context: dict
            context entries from all modalities
        """
        rec, context = {}, {}
        for (src, tgt), (mod_rec, mod_context) in rec_with_context.items():
            rec[src, tgt] = mod_rec
            prefixed_mod_context = {
                f"{src}_to_{tgt}_{context_key}": value
                for context_key, value in mod_context.items()
            }
            context.update(prefixed_mod_context)
        return rec, context
    
    def transfer(self, inputs, lengths, src, tgt=None):
        """Transfers representations from one modality to another (e.g. speech to text,
        text to speech, graphemes to phonemes, etc)
        
        Arguments
        ---------
        inputs: torch.Tensor
            A str -> tensor dictionary representing inputs in all available / relevant
            modalities
        src: str
            the source modality from which the transfer will be made
        tgt: str|list
            the target(s) to which the transfer will be performed
        

        Returns
        -------
        rec: dict
            a str -> tensor dictinaries with reconstructions in all applicable modalities
        latents: dict
            latent representations
        alignments: dict
            attention alignments, for each modality
        enc_out: dict
            raw encoder outputs
        out_context: dict
            the output context
        """
        # NOTE: Support for auxiliary modalities will need to be added later

        # Isolate the source modality
        src_inputs = {src: inputs[src]}
        src_lengths = {src: lengths[src]}
        # Find latents
        latents, alignments, enc_out, lengths_latent, _ = self.latent(src_inputs, src_lengths)
        latents = self.mask_latents(latents, lengths_latent)

        # Reconstruct
        decode_length = (
            lengths_latent[self.anchor_name]
            if self.anchor_name in lengths_latent
            else lengths_latent[src]
        )
        rec, out_context = self.decode_single(latents[src], decode_length, tgt=tgt)

        return rec, latents, alignments, enc_out, out_context


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
        rec_with_context = {
            key: self.modalities[key].decode(
                modality_latent,
                lengths=lengths.get(self.anchor_name) if lengths is not None else None,
                context=context)
            for key, modality_latent in latent.items()
        }
        return self._extract_context(rec_with_context)
    
    def train_step(self, inputs, lengths=None, context=None, transfer=False):
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

        Returns
        -------
        latents: torch.Tensor
            latent representations
        alignments: torch.Tensor
            alignment matrices
        enc_out: torch.Tensor
            raw encoder outputs
        rec: torch.Tensor
            direct reconstructions
        transfer_rec: torch.Tensor
            cross-reconstructions, if requested. If transfer=False,
            transfers will not be attempted
        out_context: dict
            the output context, from all decoders        
        latents_raw: dict
            raw, unmasked latent representations
        length_preds: dict
            length predictions
        lengths_input: dict
            input lengths
        lengths_latent: dict
            latent space lengths
        lengths_dec: dict
            lengths to be passed to decoders
            
            During training, each is set to the anchor latent length
            During evaluation, each is set to the predicted length
        """
        latents_raw, alignments, enc_out, lengths_latent, lengths_input = self.latent(
            inputs,
            lengths,
            context
        )
        
        length_preds = self.train_lengths(latents_raw, lengths)
        
        if self.training:            
            lengths_dec = {
                key: lengths_latent[self.anchor_name]
                for key in lengths
            }
        else:
            lengths_dec = {
                key: self.length_predictor.to_lengths(pred)
                for key, pred in length_preds.items()}

        latents = self.mask_latents(latents_raw, lengths_dec)
        
        rec, out_context = self.decode_multiple(latents, lengths_dec, context)
        if transfer:
            transfer_rec, transfer_context = self.cross_decode(latents, lengths_dec, context)
            out_context.update(transfer_context)
        else:
            transfer_rec = None
        
        return TrainStepOutput(
            latents=latents,
            alignments=alignments,
            enc_out=enc_out,
            rec=rec,
            transfer_rec=transfer_rec,
            out_context=out_context,
            latents_raw=latents_raw,
            length_preds=length_preds,
            lengths_input=lengths_input,
            lengths_latent=lengths_latent,
            lengths_dec=lengths_dec
        )
    
    def cross_decode(self, latents, length, context=None):
        """Decodes multipe latents into multiple modalities, useful during
        training
        
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
        rec: dict
            reconstructions, for each modality
        context: dict
            the output context
        
        """
        rec_with_context = {
            (src, tgt): self.modalities[tgt].decode(
                latents[src], length[self.anchor_name], context=context)
            for src, tgt in self.cross_transfers
        }
        return self._extract_context_cross(rec_with_context)
    
    def predict_lengths(self, latents):
        """Predicts relative lengths of latent representations
        
        Arguments
        ---------
        latents: dict
            a str -> tensor dictionary of latent representations
        
        Results
        -------
        lengths: dict
            a str -> tensor dictionary where each element is a 1D
            tensor of predicted lengths
        """
        return {
            key: self.length_predictor.lengths(latent)
            for key, latent in latents.items()
        }
    
    def train_lengths(self, latents, lengths):
        """Runs the training pass for length prediction

        Arguments
        ---------
        latents: dict
            a str -> tensor dictionary of latent representations
        
        latents: dict
            a str -> tensor dictionary of relative lengths

        Results
        -------
        lengths: dict
            a str -> tensor dictionary where each element is a 1D
            tensor of predicted lengths

        """
        anchor_length = lengths[self.anchor_name]
        return {
            key: self.length_predictor(latent, anchor_length)
            for key, latent in latents.items()
        }
    
    def mask_latents(self, latents, lengths=None):
        if lengths is not None and self.anchor in lengths:
            latents_masked = self._mask_latents_anchor(latents, lengths)
        else:
            latents_masked = self._mask_latents_predicted(latents)
        return latents_masked            
    
    def _mask_latents_predicted(self, latents):
        """Masks out latent position using predictions
        
        Arguments
        ---------
        latents: dict
            a str -> tensor dictionary of latent representations

        Returns
        -------
        result: dict
            masked-out latents
        """
        lengths = self.predict_lengths(latents)
        return {
            key: mask_out(latent, lengths[key])
            for key, latent in latents.items()
        }
    
    def _mask_latents_anchor(self, latents, lengths):
        """Masks out latent position using predictions
        
        Arguments
        ---------
        latents: dict
            a str -> tensor dictionary of latent representations
        lengths: dict
            relative lengths, per modality

        Returns
        -------
        result: dict
            masked-out latents
        """
        length = lengths[self.anchor_name]
        return {
            key: mask_out(latent, length)
            for key, latent in latents.items()
        }
    

TrainStepOutput = namedtuple(
    "TrainStepOutput",
    [
        "latents",
        "alignments",
        "enc_out",
        "rec",
        "transfer_rec",
        "out_context",
        "latents_raw",
        "length_preds",
        "lengths_input",
        "lengths_latent",
        "lengths_dec"
    ]
    
)


def mask_out(x, lengths):
    """Zeroes out positions in the specified tensor extending
    beyond provided relative lengths
    
    Arguments
    ---------
    x: torch.Tensor
        a tensor
    lengths: torch.Tensor
        relative lengths
    """
    max_len = x.size(1)
    mask = length_to_mask(lengths, max_len)
    mask = mask.unsqueeze(-1)
    return (mask * x)


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
        return self.module(x, lengths=lengths)


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

class OuputContextCallWrapper(nn.Module):
    """A wrapper that adds a empty context to the output. This is needed
    to ensure that scriptability is not compromised"""
    def __init__(self, module):
        self.module = module
    
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
        context: dict
            an empty context
        """        
        out = self.module(x, lengths, context)
        return out, {}

def wrap_module(module, needs_context=False):
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

    outputs_context = getattr(module, "outputs_context", False)
    module = wrapper(module)
    if needs_context and not outputs_context:
        module = OuputContextCallWrapper(module)
    return module
    


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
    
    mod_type: str
        the modality type

    mod_family: str
        the modality family (e.g. token_sequence, spectrogram, etc).
        MAdmixture itself does not use this property, but it can be
        useful during evaluation to determine the type of report to
        generate, when rendering a UI, etc
        

    """
    def __init__(self, name, encoder, decoder, aligner=None, mod_type=ModalityType.PRIMARY, mod_family="token_sequence"):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.aligner = aligner
        self.encoder = wrap_module(encoder)
        self.decoder = wrap_module(decoder, needs_context=True)
        if isinstance(mod_type, str):
            mod_type = ModalityType(mod_type)
        self.mod_type = mod_type
        self.mod_family = mod_family

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
        # type: (Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]
        return self.decoder(latent, lengths, context)


class Aligner(nn.Module):
    """A base class for auxiliary modules that align encoded outputs in
    a given modality to the latent space"""

    def __init__(self, eos_marker=None) -> None:
        super().__init__()
        self.eos_marker = eos_marker
    
    def forward(self, key, enc_out, lengths, anchor=None, anchor_scale=None, eos_mark=None):
        """Computes alignments

        Arguments
        ---------
        key: str
            the modality key
        enc_out: torch.Tensor
            raw encoded states
        length: torch.Tensor
            relative lengths
        anchor: str
            the modality to which outputs will be "anchored".
            If prvided, masking of outputs will be done
            based on anchor lengths
        anchor_scale: float
            the scaling factor for the anchor modality
        eos_mark: torch.nn.Module
            a module that edds an explicit learned end-of-sequence
            mark to the sequence to help with length detection


        Returns
        -------
        output: torch.Tensor
            aligned outputs
        alignment: torch.Tensor
            the alignment matrix from the attention module
        
        """
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
    scale: float
        the scaling factor
    kernel_size: int
        the kernel size for the compressor
    smoothener_kernel_size: int
        the kernel size for the aditional non-strieded "smoothening"
        layer applied after compression
    """

    def __init__(
            self,
            dim,
            in_dim=None,
            nhead=1,
            dropout=0.,
            scale=None,
            kernel_size=None,
            smoothener_kernel_size=5
        ):
        super().__init__()
        if in_dim is None:
            in_dim = dim
        self.in_dim = in_dim
        self.attn = MultiheadAttention(
            nhead=nhead,
            d_model=dim,
            dropout=dropout
        )
        self.in_norm = LayerNorm(
            input_shape=[None, None, dim]
        )
        self.out_norm = LayerNorm(
            input_shape=[None, None, dim]
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
                kernel_size = stride
            self.compressor = Conv1d(
                in_channels=in_dim,
                out_channels=dim,
                stride=stride,
                kernel_size=kernel_size,
                padding="same"
            )
        if self.scale != 1:
            self.smoothener = Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=smoothener_kernel_size,
                padding="same"
            )
        else:
            self.smoothener = nn.Identity()
        
        # Rescale the feature dimension only using a convolutional layer, if necessary
        if in_dim != dim:
            self.feature_scale = Conv1d(
                in_channels=in_dim, out_channels=dim, padding="same", kernel_size=1)
        else:
            self.feature_scale = nn.Identity()
        self.pos_emb = PositionalEncoding(
            input_size=dim
        )

    def get_queries(self, enc_out, length):
        """Upsamples or downsamples raw encoder outputs
        before applying attention   
        
        """
        enc_out_scaled = self.compressor(enc_out)
        enc_out_scaled = self.smoothener(enc_out_scaled)
        enc_out_scaled = self.out_norm(enc_out_scaled)
        return adjust_dim(enc_out_scaled, 1, length)

    def forward(self, key, enc_out, lengths, anchor=None, anchor_scale=None, eos_mark=None):
        """Computes alignments using an attention module

        Arguments
        ---------
        key: str
            the modality key
        enc_out: torch.Tensor
            raw encoded states
        length: torch.Tensor
            relative lengths
        anchor: str
            the modality to which outputs will be "anchored".
            If prvided, masking of outputs will be done
            based on anchor lengths
        anchor_scale: float
            the scaling factor for the anchor modality
        eos_mark: torch.nn.Module
            a module that edds an explicit learned end-of-sequence
            mark to the sequence to help with length detection

        Returns
        -------
        output: torch.Tensor
            aligned outputs
        alignment: torch.Tensor
            the alignment matrices from the attention module
        lengths_latent: torch.Tensor
            the lengths of latent representations
        lengths_input: torch.Tensor
            the precise lengths of inputs, in the alignment matrix
        """
        if anchor is None:
            anchor = key
        if anchor_scale is None:
            anchor_scale = self.scale
        mod_enc_out = enc_out[key]
        mod_length = lengths[key]
        anchor_length = lengths[anchor]
        anchor_max_len = int(round(enc_out[anchor].size(1) * anchor_scale))
        
        queries = self.get_queries(mod_enc_out, anchor_max_len)
        queries_max_len = queries.size(1)

        anchor_length_queries_abs = (anchor_length * queries_max_len).round().int()

        queries_mask = length_to_mask(
            anchor_length_queries_abs, queries_max_len).unsqueeze(-1)
        masked_queries = queries * queries_mask
        
        mod_enc_out_scaled = self.feature_scale(mod_enc_out)
        mod_enc_out_scaled = self.in_norm(mod_enc_out_scaled)
        pos_embs = self.pos_emb(mod_enc_out_scaled)
        mod_enc_out_scaled += pos_embs

        mod_enc_out_scaled_max_len = mod_enc_out_scaled.size(1)
        mod_enc_out_scaled_length_abs = (
            mod_length * mod_enc_out_scaled_max_len).round().int()
        if eos_mark is not None:
            mod_enc_out_scaled, mod_enc_out_scaled_length_abs = eos_mark(
                mod_enc_out_scaled, mod_enc_out_scaled_length_abs)
            mod_enc_out_scaled_max_len = mod_enc_out_scaled.size(1)

        out_mask = length_to_mask(
            mod_enc_out_scaled_length_abs, mod_enc_out_scaled_max_len).unsqueeze(-1)
        
        attn_mask = self._get_attention_mask(out_mask, queries_mask)
        
        output, alignment = self.attn(
            query=masked_queries,
            key=mod_enc_out_scaled,
            value=mod_enc_out_scaled,
            key_padding_mask=~(out_mask.bool().squeeze(-1)),
            attn_mask=attn_mask,
        )
        return output, alignment, anchor_length_queries_abs, mod_enc_out_scaled_length_abs
    
    def _get_attention_mask(self, in_mask, out_mask):
        """Computes the attention mask
        
        Arguments
        ---------
        in_mask: torch.Tensor
            the input mask
        out_mask: torch.Tensor
            the output mask

        Returns
        -------
        attn_mask: torch.Tensor
            the attention mask
        """
        batch_size, in_max_len, _ = in_mask.shape
        _, out_max_len, _ = out_mask.shape
        attn_mask = torch.zeros(batch_size, out_max_len, in_max_len).to(
            in_mask.device)
        out_mask_attn = ~out_mask.squeeze().bool()
        in_mask_attn = ~in_mask.squeeze().bool()
        attn_mask[out_mask_attn.unsqueeze(-1).repeat(1, 1, in_max_len)] = 1
        attn_mask[in_mask_attn.unsqueeze(1).repeat(1, out_max_len, 1)] = 1
        return attn_mask


class LengthPredictor(nn.Module):
    """A superclass for sequence end detection
    modules that predict the length of a sequence
    given a raw latent space
    """
    def forward(self, latent):
        """Computes the forward pass
        Arguments
        ---------
        latent: torch.Tensor
            the latent space

        Returns
        -------
        result: torch.Tensor
            a raw representation of sequence ends,
            for training, loss-dependent        
        """
        raise NotImplementedError()
    
    def to_lengths(self, length_pred):
        """Predicts absolute latent lengths from raw output
        
        Arguments
        ---------
        latent: torch.Tensor
            the latent space

        Returns
        -------
        result: torch.Tensor
            a 1-D tensor of predicted lengths
        """
        raise NotImplementedError()
    

    def lengths(self, latent):
        """Predicts relative lengths from raw output
        
        Arguments
        ---------
        latent: torch.Tensor
            the latent space

        Returns
        -------
        result: torch.Tensor
            a 1-D tensor of predicted lengths
        """
        length_pred = self(latent)
        return self.to_lengths(length_pred)



class LinearLengthPredictor(LengthPredictor):
    """A sequence end detector implemented using
    a simple linear layer
    
    Arguments
    ---------
    latent_size: int
        the latent space dimension"""
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.lin = Linear(
            input_size=latent_size,
            n_neurons=1
        )
        self.act = nn.Softmax(dim=1)

    def forward(self, latent, length=None):
        """Computes the forward pass
        Arguments
        ---------
        latent: torch.Tensor
            the latent space
        length: torch.Tensor
            the relative length (for masking)

        Returns
        -------
        result: torch.Tensor
            a raw representation of sequence ends,
            for training, loss-dependent        
        """        
        x = latent
        x = self.lin(x)
        x = x.squeeze(-1)
        if length is not None:
            max_len = latent.size(1)
            mask = length_to_mask(length * max_len, max_len)
            x *= mask
        x = self.act(x)
        return x
    
    def to_lengths(self, length_pred):
        """Predict absolute lengths
        
        Arguments
        ---------
        length_pred: torch.Tensor
            the raw module output

        Returns
        -------
        result: torch.Tensor
            a 1-D tensor of predicted lengths
        """
        return length_pred.argmax(dim=-1).clamp(1.).round().int()

class GateLengthPredictor(LengthPredictor):
    """A sequence end detector implemented using
    a simple linear layer with a sigmoid, outputting
    independent probabilities of any given 
    
    Arguments
    ---------
    latent_size: int
        the latent space dimension"""
    def __init__(self, latent_size, threshold=0.5):
        super().__init__()
        self.latent_size = latent_size
        self.lin = Linear(
            input_size=latent_size,
            n_neurons=1
        )
        self.act = nn.Sigmoid()
        self.threshold = threshold

    def forward(self, latent, length=None):
        """Computes the forward pass
        Arguments
        ---------
        latent: torch.Tensor
            the latent space
        length: torch.Tensor
            the relative length (for masking)

        Returns
        -------
        result: torch.Tensor
            a raw representation of sequence ends,
            for training, loss-dependent        
        """        
        x = latent
        x = self.lin(x)
        x = x.squeeze(-1)
        x = self.act(x)
        return x
    
    def to_lengths(self, length_pred):
        """Predict absolute lengths
        
        Arguments
        ---------
        length_pred: torch.Tensor
            the raw module output

        Returns
        -------
        result: torch.Tensor
            a 1-D tensor of predicted lengths
        """
        return (length_pred > self.threshold).int().argmax(dim=-1).clamp(1)


class EndOfSequenceMarker(nn.Module):
    """A module that adds EOS markers to a sequence, intended
    to be used for encoder outputs to facilitate length
    prediction
    
    Arguments
    ---------
    feature_size: int
        the size of the feature dimension
    marker_type: str
        "fixed" for a fixed pattern
        "learned" for a learned parameter
    length_mode: str
        "absolute": absolute (integer) lengths
        "relative": relative lengths
    """
    def __init__(
            self,
            feature_size,
            marker_type="fixed",
            length_mode="absolute"
        ):
        super().__init__()
        self.feature_size = feature_size
        self.length_mode = length_mode
        if marker_type == "fixed":
            marker = self._get_fixed_marker()  
            self.register_buffer("marker", marker)
        elif marker_type == "learned":
            self.marker = nn.Parameter(torch.randn(feature_size))
        else:
            raise ValueError(f"Invalid marker type {marker_type}")
    
    def _get_fixed_marker(self):
        """Initializes a fixed marker in a pattern of alternative
        ones and zeros
        
        Returns
        -------
        marker: torch.Tensor
            the marker"""
        marker = torch.zeros(self.feature_size)
        marker[::2] = 1.
        return marker

    def forward(self, x, length):
        """Adds an end-of-sequence marker to the end of the sequence

        Arguments
        ---------
        x: torch.Tensor
            a tensor (e.g. encoder outputs)
        length: torch.Tensor
            a length tensor (absolute or relative)

        Returns
        -------
        x_eos: torch.Tensor
            x, with the marker added
        length_eos: torch.Tensor
            the new lengths
        """
        batch_size = x.size(0)
        marker = self.marker[None, None, ...].expand(batch_size, 1, self.feature_size)
        marker_length = torch.ones(batch_size, device=x.device)
        x_eos, length_eos = concat_padded_features(
            [x, marker],
            [length, marker_length],
            length_mode=self.length_mode
        )
        return x_eos, length_eos
