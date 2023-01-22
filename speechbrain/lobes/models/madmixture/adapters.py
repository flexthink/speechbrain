"""
Adapter/wrapper classes for other existing models to be used within
the MAdmixture model

Authors
 * Artem Ploujnikov 2023
"""
from torch import nn
from speechbrain.lobes.models import Tacotron2

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