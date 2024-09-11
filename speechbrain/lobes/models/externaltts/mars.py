"""
A wrapper for the open-source implementation of Mars TTS

https://huggingface.co/spaces/k2-fsa/text-to-speech

Authors
 * Artem Ploujnikov 2024
"""
from torch import nn
from .common import TTSInferenceResult
from speechbrain.utils.data_utils import batch_pad_right
from speechbrain.utils.data_utils import undo_padding
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
        offset=0.3,
        sample_rate=24000,
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
        self.offset = offset
        self.sample_rate = sample_rate

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
        if not isinstance(spk_text, list):
            spk_wav = [spk_wav] * len(text)
            spk_text = [spk_text] * len(text)
        else:
            spk_wav_data, spk_wav_lengths = spk_wav
            spk_wav = undo_padding(spk_wav_data, spk_wav_lengths)

        results = [
            self.model.tts(
                sample,
                item_spk_wav,
                item_spk_text,
                cfg=self.cfg
            )
            for sample, item_spk_wav, item_spk_text
            in zip(text, spk_wav, spk_text)
        ]
        wav, length = batch_pad_right(
            [item_wav for _, item_wav in results]
        )
        tokens, _ = batch_pad_right(
            [item_tokens for item_tokens, _ in results]
        )
        device = next(self.parameters()).device
        wav = wav.to(device)
        length = length.to(device)
        tokens = tokens.to(device)

        offset_frames = int(self.offset * self.sample_rate)
        wav_len = wav.size(1)
        wav_new_len = wav_len - offset_frames
        length = (length * wav_len - offset_frames) / wav_new_len
        wav = wav[:, offset_frames:]

        return TTSInferenceResult(
            wav=wav,
            length=length,
            tokens=tokens
        )
