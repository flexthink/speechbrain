"""
Adapter/wrapper classes for other existing models to be used within
the MAdmixture model

Authors
 * Artem Ploujnikov 2023
"""
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
        raw_lengths = lengths * latent.size(1)
        mel_outputs, gate_outputs, alignments = self.decoder(
            memory=latent,
            decoder_inputs=context[self.decoder_input_key].transpose(-1, -2),
            memory_lengths=raw_lengths
        )
        return mel_outputs.transpose(-1, -2)
    

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
    
    act: nn.Module
        the final activation layer to use (softmax by default)
    """
    def __init__(self, rnn, input_key, act=None, out_dim=None):
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
        self.out_dim = out_dim
        if act is None:
            act = Softmax(apply_log=True)
        self.act = act

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
        input_value = context[self.input_key]
        #TODO: Add support for the default dummy value
        output, _ = self.rnn(input_value, latent, wav_len=lengths)
        output = self.lin_out(output)
        output = self.act(output)
        return output
    