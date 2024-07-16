"""
A wrapper for the open-source implementation of Mars TTS

https://huggingface.co/spaces/k2-fsa/text-to-speech

Authors
 * Artem Ploujnikov 2024
"""
from torch import nn
from .common import TTSInferenceResult
from speechbrain.utils.data_utils import batch_pad_right
import torch


DEFAULT_SOURCE = 'Camb-ai/mars5-tts'
DEFAULT_MODEL = 'mars5_english'


class Mars(nn.Module):
    def __init__(
        self,
        source=None,
        model=None,
        deep_clone=True,
        rep_penalty_window=100,
        top_k=100,
        temperature=0.7,
        freq_penalty=3,
    ):
        super().__init__()
        if source is None:
            source = DEFAULT_SOURCE
        if model is None:
            model = DEFAULT_MODEL
        self.model, self.config_class = torch.hub.load(
            source, model, trust_repo=True
        )
        self.cfg = self.config_class(
            deep_clone=deep_clone,
            rep_penalty_window=rep_penalty_window,
            top_k=top_k,
            temperature=temperature,
            freq_penalty=freq_penalty,
        )
    
    def forward(self, text, spk=None, language=None):
        """Performs inference

        Arguments
        ---------
        text : list
            A list of strings (raw text)
        spk : str | tuple
            The speaker identity - a pre-defined preset or a (wav, text)
            tupple
        language : str
            The language code (if used in a multilingual context)


        Returns
        -------
        wav : torch.Tensor
            A tensor of raw waveforms
        length : torch.Tensor
            Relative lengths
        tokens : torch.Tensor
            Raw tokens
        """
        spk_wav, spk_text = spk
        results = [
            self.model.tts(
                sample,
                spk_wav,
                spk_text,
                cfg=self.cfg
            )
            for sample in text
        ]
        wav, length = batch_pad_right(
            [item_wav for _, item_wav in results]
        )
        tokens, _ = batch_pad_right(
            [item_tokens for item_tokens, _ in results]
        )
        return TTSInferenceResult(
            wav=wav,
            length=length,
            tokens=tokens
        )