"""
A wrapper for the implementation of Bark

https://github.com/Plachtaa/VALL-E-X

Authors
 * Artem Ploujnikov 2024
"""

import torch
from torch import nn
from .common import TTSInferenceResult


try:
    from transformers import AutoProcessor, AutoModel
except ImportError:
    err_msg = (
        "The optional dependency transformers must be installed to use Bark\n"
    )
    err_msg += "Install using `pip install transformers`.\n"
    raise ImportError(err_msg)


HF_MODEL_BARK = "suno/bark"
DEFAULT_VOICE_PRESET = "v2/{language}_speaker_1"


class Bark(nn.Module):
    """A wrapper for the Bark TTS model

    Arguments
    ---------
    source : str, optional
        The HuggingFace hub for the model source
    savedir : str
        The path where the model will be saved
    default_spk : str
        The default speaker identifier
        It may contain a {language} placeholder, which,
        if encountered, will be replaced with the language
        identifier
    default_language : str
        The default language code
    device : str
        The device to use
    """
    def __init__(
        self,
        source=None,
        savedir=None,
        default_spk=None,
        default_language="en",
        device="cpu",
    ):
        super().__init__()
        if source is None:
            source = HF_MODEL_BARK
        if default_spk is None:
            default_spk = DEFAULT_VOICE_PRESET
        self.source = source
        self.savedir = savedir
        self.processor = AutoProcessor.from_pretrained(source)
        self.model = AutoModel.from_pretrained(source)
        self.default_spk = default_spk
        self.default_language = default_language
        self.device = device

    def forward(self, text, spk=None, language=None):
        """Performs inference

        Arguments
        ---------
        text : list
            A list of strings (raw text)
        spk : str
            The speaker identity
        language : str
            The language code (if used in a multilingual context)

        Returns
        -------
        wav : torch.Tensor
            Returns the synthesized waveform
        """
        if spk is not None:
            spk = self.default_spk
        if language is None:
            language = self.default_language
        spk = spk.format(language=language)
        inputs = self.processor(
            text,
            voice_preset=spk
        )
        inputs = self._to_device(inputs)
        wav, length_abs = self.model.generate(
            **inputs,
            do_sample=True,
            return_output_lengths=True,
        )
        length = torch.tensor(length_abs, device=self.device) / wav.size(1)
        return TTSInferenceResult(
            wav=wav,
            length=length,
            tokens=None
        )

    def _to_device(self, inputs):
        if torch.is_tensor(inputs):
            result = inputs.to(self.device)
        elif hasattr(inputs, "items"):
            result = {
                key: self._to_device(value)
                for key, value in inputs.items()
            }
        else:
            result = inputs
        return result

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)
