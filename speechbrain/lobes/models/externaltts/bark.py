"""
A wrapper for the implementation of VALL-E X

https://github.com/Plachtaa/VALL-E-X

Authors
 * Artem Ploujnikov 2024
"""

from torch import nn

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
        identifier"""
    def __init__(
        self,
        source=None,
        savedir=None,
        default_spk=None,
        default_language="en",
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
        wav = self.model.generate(
            **inputs,
            do_sample=True
        )
        return wav
