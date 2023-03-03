"""
Adapter/wrapper classes for other existing models to be used within
the MAdmixture model

Authors
 * Artem Ploujnikov 2023
"""
import torch
from torch import nn
from speechbrain.lobes.models import Tacotron2
from speechbrain.nnet.activations import Softmax

class TacotronDecoder(nn.Module):
    """An audio / spectrogram decoder that uses the speechbrain Tacotron implementation
    to decode an audio signal from a latent space
    
    Arguments
    ---------
    decoder_input_key
        the key within the context to find decoder inputs for teacher forcing
    """
    outputs_context = True
    
    def __init__(self, decoder_input_key="spec", **kwargs):
        super().__init__()
        self.decoder_input_key = decoder_input_key
        self.decoder = Tacotron2.Decoder(
            **kwargs
        )

    def forward(self, latent, lengths, context):
        """Decodes latent representations via Tacotron

        Arguments
        ---------
        latent: torch.Tensor
            the latent representation
        
        lengths: torch.Tensor
            relative lengths

        context: torch.Tensor
            a context dictionary (from MadMixture)

        """
        max_len = lengths.max().item()
        mel_lengths = None
        latent_cut = latent[:, :max_len, :]
        if context is not None and self.decoder_input_key in context:        
            mel_outputs, gate_outputs, alignments = self.decoder(
                memory=latent_cut,
                decoder_inputs=context[self.decoder_input_key].transpose(-1, -2),
                memory_lengths=lengths
            )
        else:
            mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
                memory=latent_cut,
                memory_lengths=lengths
            )
        out_context = {
            "gate_outputs": gate_outputs,
            "decoder_alignments": alignments,
        }
        if mel_lengths is not None:
            out_context["mel_lengths"] = mel_lengths
        return mel_outputs.transpose(-1, -2), out_context

class RNNEncoder(nn.Module):
    """An encoder adapter for RNN modules (LSTM, GRU, etc)
    
    Arguments
    ---------
    rnn: torch.Module
        a module compatible with speechbrain.nnet.RNN.*
    """
    def __init__(self, rnn, emb=None):
        super().__init__()
        self.rnn = rnn
        if emb is None:
            emb = nn.Identity()
        self.emb = emb

    def forward(self, input, lengths):
        """Performs the encoding forward pass. Hidden
        states from the RNN are discarded
        
        Arguments
        ---------
        input: torch.Tensor
            the input features

        Returns
        -------
        output: torch.Tensor
            the encoded inputs
        """
        emb = self.emb(input)
        output, _ = self.rnn(emb, lengths=lengths)
        return output
    
class RNNDecoder(nn.Module):
    """An encoder adapter for RNN modules (LSTM, GRU, etc)
    
    Arguments
    ---------
    latent_size: int
        the latent space dimension

    rnn: torch.Module
        an RNN module with an interface compatible to
        speechbrain.nnet.RNN.AttentionalRNNDecoder

    input_key: str
        the context key corresponding to the RNN input
        sequence
    
    out_dim: int
        the output dimension
        If specified, a linear layer will be added to
        output the correct shape
    
    emb: nn.Module
        the output sequenece imbedding module
    
    act: nn.Module
        the final activation layer to use (softmax by default)
    
    bos_index: int
        the index of the BOS token (needed during inference)
    """
    
    outputs_context = True

    def __init__(self, latent_size, rnn, input_key, emb, act=None, out_dim=None, bos_index=0):
        super().__init__()
        self.rnn = rnn
        if out_dim is None:
            self.lin_out = nn.Identity()
        else:                
            self.lin_out = nn.Linear(
                in_features=rnn.hidden_size,
                out_features=out_dim,
                bias=False
            )
        self.input_key = input_key
        self.emb = emb
        self.out_dim = out_dim
        if act is None:
            act = Softmax(apply_log=True)
        self.act = act
        self.register_buffer(
            "latent_bos", self._get_latent_bos(
                size=latent_size
            )
        )
        self.bos_index = 1

    def _get_latent_bos(self, size):
        marker = torch.ones(size)
        marker[::2] = 0
        return marker

    def forward(self, latent, lengths, context):
        """Performs the decoding forward pass
        
        Arguments
        ---------
        latent: torch.Tensor
            The latent representation to be decoded

        lengths: torch.Tensor
            Relative lengths

        context: dict
            A str -> Tensor context from MadMixture
            the input_key will be fed as ground truth
            encoder outputs during training
        """
        batch_size, max_len, _ = latent.shape
        latent_length = lengths.float() / max_len

        if self.training:            
            input_value = context[self.input_key]
            latent, latent_length = self._add_latent_bos(latent, latent_length)
        else:
            input_value = self._get_dummy_sequence(
                batch_size, device=latent.device)
        #TODO: Add support for the default dummy value
        output, alignments = self.rnn(
            input_value, latent, wav_len=latent_length)
        output = self.lin_out(output)
        output = self.act(output)
        out_context = {"alignments": alignments}
        return output, out_context
    
    def _add_latent_bos(self, latent, latent_length):
        batch_size, max_len, latent_size = latent.shape
        latent_length_abs = latent_length * max_len
        latent_length_bos = (
            (latent_length_abs + 1) / (max_len + 1)
        )
        bos_seq = self.latent_bos[None, None, :].expand(
            batch_size, 1, latent_size
        )
        latent_bos = torch.concat(
            (bos_seq, latent),
            dim=1
        )
        return latent_bos, latent_length_bos
    
    def _get_dummy_sequence(self, batch_size, device):
        seq = torch.tensor([[self.bos_index]], device=device)
        seq = seq.repeat(batch_size, 1)
        return self.emb(seq)
