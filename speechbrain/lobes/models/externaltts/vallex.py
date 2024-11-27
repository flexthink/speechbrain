"""
A wrapper for the open-source implementation of VALL-E X

https://github.com/Plachtaa/VALL-E-X

Authors
 * Artem Ploujnikov 2024
"""

from speechbrain.utils.superpowers import run_shell
from pathlib import Path
from .common import InstallCommandError, TTSInferenceResult
from importlib import import_module
from speechbrain.utils.data_utils import batch_pad_right
from speechbrain.utils.data_utils import undo_padding
from speechbrain.dataio.dataio import clean_padding
from encodec import EncodecModel
from torch import nn
import logging
import pathlib
import platform
import shlex
import sys
import torch
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "https://github.com/Plachtaa/VALL-E-X"
DEFAULT_TOKENIZER = "utils/g2p/bpe_69.json"
DEFAULT_CHECKPOINT = "vallex-checkpoint.pt"


NUM_LAYERS = 12
NUM_HEAD = 16
N_DIM = 1024
PREFIX_MODE = 1
NUM_QUANTIZERS = 8


class VALLEX(nn.Module):
    """A Vall-E X Wrapper
    
    Arguments
    ---------
    source : str, optional
        The source code repository
    savedir : str, optional
        The path where the model will be saved
    tokenizer_path : str | path-like, optional
        The path to the tokenizer
    ckpt_path : str | path-like, optional
        The path to the checkpoint to be loaded
    language : str, optional
        The language identifier
    top_k : int
        The number of highest probability tokens to keep for top-k-filtering. Default to -100.
    best_of : int, optional
        The number of samples from which the best will be selected
    n_dim : int, optional
        The model dimension
    num_head : int, optional
        The number of attention heads
    num_layers : int, optional
        The number of layers 
    num_quantizers : int, optional
        The number of quantizers to be used
    prefix_mode : int
        The prefix mode to be used
    device : str | torch.Device
        The target device identifier  
    """
    def __init__(
        self,
        source=None,
        savedir=None,
        ckpt_path=None,
        tokenizer_path=None,
        lanugage="en",
        top_k=-100,
        best_of=5,
        n_dim=1024,
        num_head=16,
        num_layers=12,
        num_quantizers=8,
        target_bandwidth=6.0,
        prefix_mode=1,
        offset=0.0,
        preset="neutral",
        sample_rate=24000,
        device="cpu"
    ):
        super().__init__()
        if source is None:
            source = DEFAULT_SOURCE
        self.source = source
        if savedir is None:
            savedir = Path("~/vallex").expanduser()
        self.savedir = Path(savedir)
        if tokenizer_path is None:
            tokenizer_path = self.savedir / DEFAULT_TOKENIZER
        self.tokenizer_path = Path(tokenizer_path)
        self.language = lanugage
        self.top_k = top_k
        self.best_of = best_of
        self.n_dim = n_dim
        self.num_head = num_head
        self.num_layers = num_layers
        self.num_quantizers = num_quantizers
        self.target_bandwidth = target_bandwidth
        self.prefix_mode = prefix_mode
        self.preset = preset
        self.device = device
        self.install()
        self.init()
        if ckpt_path is None:
            ckpt_path = self.savedir / "checkpoints" / DEFAULT_CHECKPOINT
        self.load_ckpt(ckpt_path)
        self.sample_rate = sample_rate
        self.offset = offset

    def install(self):
        if self.is_installed():
            logger.info("VALL-E X is already installed")
            return

        if not self.savedir.exists():
            # Clone the Git repository
            logger.info("Cloning the repo")
            cmd = shlex.join(
                [
                    "git",
                    "-C",
                    str(self.savedir.parent),
                    "clone",
                    self.source,
                    self.savedir.name
                ]
            )
            out, err, code = run_shell(cmd)
            if code != 0:
                raise InstallCommandError(
                    f"Unable to clone {self.source}, please install VALL-E X manually",
                    code, out, err
                )
            # Install dependencies
            logger.info("Installing dependencies")
            reqs_path = self.savedir / "requirements.txt"
            cmd = shlex.join(
                [
                    "pip",
                    "install",
                    "-r",
                    str(reqs_path)
                ]
            )
            out, err, code = run_shell(cmd)
            if code != 0:
                raise InstallCommandError(
                    "Unable to install requirements, please install VALL-E X manually",
                    code, out, err
                )

    def init(self):
        logger.info("VALL-E X: Adding %s to PYTHONPATH", self.savedir)
        sys.path.append(str(self.savedir))

        # Tokenizer (G2P)
        utils_g2p = import_module("utils.g2p")
        self.tokenizer = utils_g2p.PhonemeBpeTokenizer(
            tokenizer_path=str(self.tokenizer_path)
        )

        # Collater
        data_collation = import_module("data.collation")
        self.collater = data_collation.get_text_token_collater()
        self.collater.pad_symbol = 0

        # Encodec (audo tokenizer)
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(self.target_bandwidth)

        # VALL-E X
        models_vallex = import_module("models.vallex")
        self.model = models_vallex.VALLE(
            self.n_dim,
            self.num_head,
            self.num_layers,
            norm_first=True,
            add_prenet=False,
            prefix_mode=self.prefix_mode,
            share_embedding=True,
            nar_scale_factor=1.0,
            prepend_bos=True,
            num_quantizers=self.num_quantizers,
        ).to(self.device)
        self.model.eval()

        # Vocoder (Vocos)
        vocos = import_module("vocos")
        self.vocos = vocos.Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(self.device)
        self._apply_platform_hack()

    def _apply_platform_hack(self):
        if platform.system().lower() == 'windows':
            pathlib.PosixPath = pathlib.WindowsPath
        else:
            pathlib.WindowsPath = pathlib.PosixPath

    def load_ckpt(self, ckpt_path):
        """Loads the specified checkpoint

        Arguments
        ---------
        ckpt_path : str | path-like
            The path to the checkpoint to load
        """
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(
            checkpoint["model"], strict=True
        )

    def is_installed(self):
        """Determines whether the library is already installed"""

        return (self.savedir / ".installed").exists()

    def get_preset_audio_prompt(self, preset):
        """Retrieves the pre-set audio prompt

        Arguments
        ---------
        present : str
            The name of the pre-set to be used"""
        path = self.savedir / "presets" / f"{preset}.npz"
        data = np.load(path)
        return (
            torch.from_numpy(data["audio_tokens"]).to(self.device),
            torch.from_numpy(data["text_tokens"]).to(self.device)
        )

    def get_waveform_audio_prompt(self, wav, text, language=None):
        """Converts an audio-text pair to a VALLE-X prompt

        Arguments
        ---------
        wav : torch.Tensor
            A raw audio waveform tensor
        text : str
            The text annotation
        language : str
            The language identifier. If omitted, the default language is used

        Results
        -------
        wav : torch.Tensor
            """
        if language is None:
            language = self.language
        lang_token = get_language_token(language)
        text_prompt = "".join(["_", lang_token, text, lang_token])
        text_tokens, _ = self.tokenizer.tokenize(text=text_prompt)
        text_tokens = torch.tensor(text_tokens, device=self.device).unsqueeze(0)
        while wav.dim() < 3:
            wav = wav.unsqueeze(0)
        audio_tokens, _ = self.encodec.encode(wav)[0]
        audio_tokens = audio_tokens.transpose(-1, -2)
        return audio_tokens, text_tokens

    def get_audio_prompts(self, spk, language, count):
        """Prepares audio prompts

        Arguments
        ---------
        spk : str|tuple, optional
            One of the identifiers supported by Vall-E X
            or a (wav, text) tuple for voice cloning

        language : str, optional
            a language identifier,
        
        count : int
            the number of items

        Returns
        -------
        audio_tokens : torch.Tensor
            The audio file
        text_tokens : torch.Tensor
            The encoded text corresponding to the prompt
        """
        multiple = False
        if spk is None:
            audio_tokens, text_tokens = self.get_preset_audio_prompt(self.preset)
        elif isinstance(spk, str):
            audio_tokens, text_tokens = self.get_preset_audio_prompt(self.preset)
        elif isinstance(spk, tuple):
            wav, text = spk
            # TODO: Fix vectorization - this is not efficient
            if isinstance(text, list):
                wav = undo_padding(wav.data, wav.lengths)
                prompts = [
                    self.get_waveform_audio_prompt(item_wav, item_text, language=language)
                    for item_wav, item_text in zip(wav, text)
                ]
                audio_tokens = [item for item, _ in prompts]
                text_tokens = [item for _, item in prompts]
                multiple = True
            else:
                audio_tokens, text_tokens = self.get_waveform_audio_prompt(wav, text, language=language)
        else:
            raise ValueError("Unsupported speaker prompt")

        if not multiple:
            audio_tokens = [audio_tokens] * count
            text_tokens = [text_tokens] * count
        
        return audio_tokens, text_tokens

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
        if language is None:
            language = self.language
        language_token = get_language_token(language)
        prompt_text = [
            "".join(["_", language_token, item, language_token])
            for item in text]
        prompt_token_lang = [
            self.tokenizer.tokenize(text=f"_{item}".strip())
            for item in prompt_text
        ]
        audio_prompt_tokens, audio_prompt_text_tokens = self.get_audio_prompts(spk, language, len(text))

        # Batch inference is not supported because of an implementation issue
        # in the underlying library
        # https://github.com/Plachtaa/VALL-E-X/issues/177
        encoded_frames_items = [
            self._inference(
                text=item,
                audio_prompt_tokens=item_audio_prompt_tokens,
                audio_prompt_text_tokens=item_audio_prompt_text_tokens,
                prompt_language=language,
                text_language=item_lang
            )
            for (item, item_lang), item_audio_prompt_tokens, item_audio_prompt_text_tokens
            in zip(prompt_token_lang, audio_prompt_tokens, audio_prompt_text_tokens)
        ]
        encoded_frames, length = batch_pad_right(encoded_frames_items)
        frames = encoded_frames.permute(2, 0, 1)
        features = self.vocos.codes_to_features(frames)
        wav = self.vocos.decode(features, bandwidth_id=torch.tensor([2], device=self.device))
        wav = clean_padding(wav, length)
        length = length.to(self.device)
        offset_frames = int(self.offset * self.sample_rate)
        wav_len = wav.size(1)
        wav_new_len = wav_len - offset_frames
        length = (length * wav_len - offset_frames) / wav_new_len
        wav = wav[:, offset_frames:]
        return TTSInferenceResult(
            wav=wav,
            length=length,
            tokens=frames.transpose(-1, -2)
        )

    def _inference(self, text, audio_prompt_tokens, audio_prompt_text_tokens, prompt_language, text_language):
        """Performs inference via Vall-E X

        Arguments
        ---------
        text : str
            The inference text
        audio_prompt_tokens : torch.Tensor
            The tokens corresponding to the audio prompt (for the speaker's voice, etc)
        audio_prompt_text_tokens : torch.Tensor
            The text tokens corresponding to the audio prompt
        prompt_language : str
            The language of the prompt

        Returns
        -------
        encoded_frames : torch.Tensor
            Speech tokens of the synthesized segment
        """
        text_tokens, text_tokens_lens = self.collater([text])
        enroll_x_lens = audio_prompt_text_tokens.shape[1]
        text_token_cat_lens = text_tokens_lens + enroll_x_lens
        text_tokens_cat = torch.cat([audio_prompt_text_tokens, text_tokens.to(self.device)], dim=-1)
        encoded_frames = self.model.inference(
            text_tokens_cat.to(self.device),
            text_token_cat_lens.to(self.device),
            audio_prompt_tokens,
            enroll_x_lens=enroll_x_lens,
            top_k=self.top_k,
            temperature=1,
            prompt_language=prompt_language,
            text_language=text_language,
            best_of=self.best_of,
        )
        return encoded_frames[0]

    def to(self, device):
        super().to(device)
        self.device = device


def get_language_token(language):
    """Computes the language token for the specified language identifier
    
    Arguments
    ---------
    language : str
        The language code
    
    Returns
    -------
    language_token : str
        The language token correspndoing to the language code"""
    return f"[{language.upper()}]"
