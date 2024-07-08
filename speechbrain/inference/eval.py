""" Specifies the inference interfaces for speech quality
evaluation, used to assess the quality/intelligibility of
text-to-speech systems

Authors:
* Artem Ploujnikov 2024
"""

from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.inference.interfaces import Pretrained
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.lobes.models.huggingface_transformers import Whisper
from speechbrain.decoders.seq2seq import S2SWhisperGreedySearch
from speechbrain.dataio.batch import PaddedBatch, undo_batch
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.dataio.dataio import write_audio
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from torch.nn import ModuleDict
from tqdm.auto import tqdm
import csv
import os
import math
import json
import torch
import torchaudio
import re
import string
import logging
import shutil
import subprocess


logger = logging.getLogger(__name__)

RE_PUNCTUATION = re.compile(
    "|".join(
        re.escape(char) for char in string.punctuation
    )
)


SpeechEvaluationResult = namedtuple(
    "SpeechEvaluationResult", ["score", "details"]
)


class SpeechEvaluator:
    """A base class for speech evaluators

    Arguments
    ---------
    sample_rate : int
        The audio sample rate this evaluator expects
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def evaluate_file(self, file_name, text=None):
        """Evaluates a single file

        Arguments
        ---------
        file_name : str|pathlib.Path
            The file name to evaluate
        text : str
            The ground truth text, if applicable

        Returns
        -------
        result: SpeechEvaluationResult
            the evaluation result
        """
        wav = self.read_audio(str(file_name)).to(self.device)
        result = self.evaluate(
            wavs=wav.unsqueeze(0),
            length=torch.ones(1).to(self.device),
            text=[text],
        )
        return SpeechEvaluationResult(
            score=result.score.item(),
            details={
                key: _unbatchify(value) for key, value in result.details.items()
            },
        )

    def evaluate_files(self, file_names, text=None):
        """Evaluates multiple files

        Arguments
        ---------
        file_names : list
            A list of files

        text : list
            File transcripts (not required for all evaluators)

        Returns
        -------
        result : list
            a list of SpeechEvaluationResult instances
        """
        if text is None:
            text = [None] * len(file_names)
        items = [
            {"wav": self.read_audio(str(file_name)), "text": item_text}
            for file_name, item_text in zip(file_names, text)
        ]
        batch = PaddedBatch(items)
        return self.evaluate(
            wavs=batch.wav.data.to(self.device),
            length=batch.wav.lengths.to(self.device),
            text=batch.text,
        )

    def read_audio(self, file_name):
        """Reads an audio file, resampling if necessary

        Arguments
        ---------
        file_name : str | path-like
            The file path

        Returns
        -------
        audio : torch.Tensor
            the audio
        """
        audio, audio_sample_rate = torchaudio.load(str(file_name))
        return self.resample(audio, audio_sample_rate)

    def evaluate(self, wavs, length, text=None, wavs_ref=None, wavs_length_ref=None, sample_rate=None):
        """Evaluates samples

        Arguments
        ---------
        wavs : torch.Tensor
            the waveforms to evaluate

        length : torch.Tensor
            relative lengths (a 1-D tensor)

        text : list
            Evaluator-specific metadata

        wavs_ref : torch.Tensor
            the reference waveforms

        wavs_length_ref
            the reference waveform lengths

        sample_rate: int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model

        Returns
        -------
        result : list
            A list of SpeechEvaluationResult objects,
            one for each sample"""
        raise NotImplementedError()

    def resample(self, audio, sample_rate=None):
        """Resamples the audio, if necessary

        Arguments
        ---------
        audio : torch.Tensor
            the audio to be resampled
        sample_rate : int
            the sample rate of the audio

        Returns
        -------
        audio : torch.Tensor
            the target audio, resampled if necessary
        """
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
        return audio


def _unbatchify(value):
    """Removes the batch dimension from the tensor. If a single
    number is returned in any shape, the function converts
    the result to a numeric value. Values that are not tensors
    are returned unmodified

    Arguments
    ---------
    value : object
        the value

    Returns
    -------
    value : object
        the value with the batch dimension removed, if applicable
    """
    if torch.is_tensor(value):
        if value.dim() == 0 or not any(dim > 1 for dim in value.shape):
            value = value.item()
        else:
            value = value.squeeze(0)
    return value


class SpeechEvaluationRegressionModel(Pretrained):
    """A pretrained wrapper for regression-based evaluaton
    models"""

    def __call__(self, wavs, length):
        return self.mods.model(wavs, length)


class RegressionModelSpeechEvaluator(SpeechEvaluator):
    """A speech evaluator that uses a regression model
    that produces a quality score (e.g. SSL fine-tuning)
    for a sample of speech

    Arguments
    ---------
    source : str
        The source model path or HuggingFace hub name
    sample_rate : int
        The audio sample rate this evaluator expects
    """

    def __init__(self, source, sample_rate=None, *args, **kwargs):
        super().__init__(sample_rate=sample_rate)
        self.model = SpeechEvaluationRegressionModel.from_hparams(
            source, *args, **kwargs
        )

    def evaluate(self, wavs, length, text=None, wavs_ref=None, length_ref=None, sample_rate=None, sample_rate_ref=None):
        """Evaluates a batch of waveforms

        Arguments
        ---------
        Arguments
        ---------
        wavs: torch.Tensor
            the waveforms to evaluate

        length: torch.Tensor
            relative lengths (a 1-D tensor)

        text : list, optional
            Ground truth text

        wavs_ref : torch.Tensor
            the reference waveforms

        length_ref : torch.Tensor
            the reference waveform lengths

        sample_rate : int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model

        sample_rate_ref : int, optional
            The sample rate of the reference samples

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        wavs = self.resample(wavs, sample_rate)
        scores = self.model(wavs, length)
        while scores.dim() > 1 and scores.size(-1) == 1:
            scores = scores.squeeze(-1)
        return SpeechEvaluationResult(score=scores, details={"score": scores})


class ASRSpeechEvaluator(SpeechEvaluator):
    def evaluate(self, wavs, length, text=None, wavs_ref=None, length_ref=None, sample_rate=None, sample_rate_ref=None):
        """Evaluates samples

        Arguments
        ---------
        wavs: torch.Tensor
            the waveforms to evaluate

        length: torch.Tensor
            relative lengths (a 1-D tensor)

        text : list, optional
            Ground truth text

        wavs_ref : torch.Tensor
            the reference waveforms

        length_ref : torch.Tensor
            the reference waveform lengths


        sample_rate : int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model

        sample_rate_ref : int, optional
            The sample rate of the reference samples

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        details = self.evaluate_samples(
            wavs=wavs,
            length=length,
            text=text,
            sample_rate=sample_rate
        )
        if wavs_ref is not None:
            details_ref = self.evaluate_samples(
                wavs=wavs_ref,
                length=length_ref,
                text=text,
                sample_rate=sample_rate_ref
            )
            details.update(
                {
                    f"{key}_ref": value
                    for key, value in details_ref.items()
                }
            )
            # Redundant: it is the same
            del details["target_ref"]
            details.update(
                self.compute_diff_rate(details, device=wavs.device)
            )

        return SpeechEvaluationResult(
            score=details["wer"],
            details=details,
        )

    def compute_diff_rate(self, details, device):
        ids = range(1, len(details["pred"]) + 1)
        wer_metric, cer_metric = init_asr_metrics()
        pred = self._replace_blanks(details["pred"])
        pred_ref = self._replace_blanks(details["pred_ref"])
        wer_metric.append(ids, pred, pred_ref)
        cer_metric.append(ids, pred, pred_ref)
        dwer = torch.tensor(
            [score["WER"] for score in wer_metric.scores],
            device=device
        )
        dcer = torch.tensor(
            [score["WER"] for score in cer_metric.scores],
            device=device
        )
        return {"dwer": dwer, "dcer": dcer}

    def _replace_blanks(self, preds):
        return [" " if item == "" else item for item in preds]


class EncoderDecoderASRSpeechEvaluator(ASRSpeechEvaluator):
    """A speech evaluator implementation based on ASR.
    Computes the Word Error Rate (WER), Character Error Rate (CER)
    and a few other metrics

    Arguments
    ---------
    sample_rate : int
        The audio sample rate this evaluator expects    
    """
    def __init__(self, source, sample_rate=None, *args, **kwargs):
        super().__init__(sample_rate=sample_rate)
        self.asr = EncoderDecoderASR.from_hparams(
            source, *args, **kwargs
        )
        self.device = next(self.asr.mods.parameters()).device

    def evaluate_samples(self, wavs, length, text, sample_rate):
        wavs = self.resample(wavs, sample_rate)
        if text is None:
            raise ValueError("This evaluator requires ground-truth text")
        predicted_words, scores, log_probs = self.transcribe_batch_with_details(
            wavs, length
        )
        ids = range(1, len(wavs) + 1)
        wer_metric, cer_metric = init_asr_metrics()
        wer_metric.append(ids, predicted_words, text)
        cer_metric.append(ids, predicted_words, text)
        wer = torch.tensor(
            [score["WER"] for score in wer_metric.scores],
            device=wavs.device
        )
        cer = torch.tensor(
            [score["WER"] for score in cer_metric.scores],
            device=wavs.device
        )
        prob_mean = log_probs.exp().mean(dim=-1)
        return {
            "wer": wer,
            "cer": cer,
            "beam_score": scores,
            "prob_mean": prob_mean,
            "pred": predicted_words,
            "target": text,
        }

    def transcribe_batch_with_details(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        predicted_words : list
            The raw ASR predictions, fully decoded
        best_scores : list
            The best scores (from beam search)
        best_log_probs : list
            The best predicted log-probabilities (from beam search)


        Returns
        -------
        predicted_words : list
            The predictions

        best_scores : torch.Tensor
            The best scores (from beam search)

        best_log_probs : torch.Tensor
            The best log-probabilities

        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.asr.encode_batch(wavs, wav_lens)
            (
                hyps,
                best_lens,
                best_scores,
                best_log_probs,
            ) = self.asr.mods.decoder(encoder_out, wav_lens)
            predicted_words = [
                self.asr.tokenizer.decode_ids(token_seq) for token_seq in hyps
            ]
        return predicted_words, best_scores, best_log_probs

    def to(self, device):
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.Device
            the target device
        """
        self.asr = self.asr.to(device)
        return self


class WhisperASRSpeechEvaluator(ASRSpeechEvaluator):
    def __init__(
        self,
        source,
        savedir=None,
        sample_rate=22050,
        bos_index=50363,
        eos_index=50257,
        min_decode_ratio=0.0,
        max_decode_ratio=1.0,
        run_opts=None,
    ):
        if run_opts is None:
            run_opts = {}
        super().__init__(sample_rate=sample_rate)
        if savedir is None:
            savedir = "."
        self.model = Whisper(
            source,
            savedir,
            sample_rate,
            freeze=True,
            freeze_encoder=True,
        )
        self.model.tokenizer.set_prefix_tokens("english", "transcribe", False)
        self.searcher = S2SWhisperGreedySearch(
            self.model,
            bos_index=bos_index,
            eos_index=eos_index,
            min_decode_ratio=min_decode_ratio,
            max_decode_ratio=max_decode_ratio,
        )
        self.searcher.set_decoder_input_tokens(
            self.model.tokenizer.prefix_tokens
        )
        device = run_opts.get(
            "device", 
            next(self.model.parameters()).device
        )
        self.to(device)

    def evaluate_samples(self, wavs, length, text, sample_rate):
        if text is None:
            raise ValueError("This evaluator requires ground-truth text")
        wavs = self.resample(wavs, sample_rate)
        enc_out = self.model.forward_encoder(
            wavs
        )
        predicted_words, _, _, _  = self.searcher(
            enc_out, length
        )
        predicted_words = self.model.tokenizer.batch_decode(
            predicted_words, skip_special_tokens=True
        )
        predicted_words = [
            self.normalize(text)
            for text in predicted_words
        ]
        ids = range(1, len(wavs) + 1)
        wer_metric, cer_metric = init_asr_metrics()
        wer_metric.append(ids, predicted_words, text)
        cer_metric.append(ids, predicted_words, text)
        wer = torch.tensor(
            [score["WER"] for score in wer_metric.scores],
            device=wavs.device
        )
        cer = torch.tensor(
            [score["WER"] for score in cer_metric.scores],
            device=wavs.device
        )
        return {
            "wer": wer,
            "cer": cer,
            "pred": predicted_words,
            "target": text,
        }

    def normalize(seflf, text):
        text = text.upper()
        text = text.strip()
        text = RE_PUNCTUATION.sub("", text)
        return text

    def to(self, device):
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.Device
            the target device
        """
        self.model = self.model.to(device)
        return self


def itemize(result):
    """Converts a single batch result into per-item results

    Arguments
    ---------
    result: SpeechEvaluationResult
        a single batch result

    Returns
    -------
    results: list
        a list of individual SpeechEvaluationResult instances"""

    return [
        SpeechEvaluationResult(
            score=result.score[idx],
            details={key: value[idx] for key, value in result.items()},
        )
        for idx in range(len(result.score))
    ]


def init_asr_metrics():
    """Initializes the WER and CER metrics

    Returns
    -------
    wer_metric : ErrorRateStats
        the Word Error Rate (WER) metric
    cer_metric : ErrorRateStats
        the Character Error Rate (CER) metric"""
    wer_metric = ErrorRateStats()
    cer_metric = ErrorRateStats(split_tokens=True)
    return wer_metric, cer_metric


class BulkSpeechEvaluator:
    def evaluate_files(self, file_names, text=None, file_names_ref=None):
        raise NotImplementedError()


class UTMOSSpeechEvaluator(BulkSpeechEvaluator):
    """An evaluation wrapper for UTMOS

    Github: https://github.com/sarulab-speech/UTMOS22
    HuggingFace: https://huggingface.co/spaces/sarulab-speech/UTMOS-demo

    Arguments
    ---------
    model_path : str | path-like
        The path where the HuggingFace repository was extracted
    output_folder : str | path-like
        The folder where results will be output
    ckpt_path : str | path-like
        The path to the checkpoint to be used
    script : str | path-like
        The path to the evaluation script, defaults to the bundled
        predict.py
    python : str | path-like, optional
        The path to the Python interpreter to be used, defaults to
        "python". Depending on the environment, it might need to be
        changed (e.g. to "python3" or an absolute path to the interpreter)
    use_python : bool
        Whether to launch the script using python. This flag will need to be
        set to False in environments where running UTMOS requires a wrapper shell
        script (e.g. to initialize a different Python virtual environment from
        the one in which SpeechBrain is running)
    tmp_folder : str | path-like, optional
        The temporary folder where files will be copied for evaluation. If
        omitted, it will be set to output_folder. This can be useful on
        compute environments that provide fast local storage (e.g. certain
        compute clusters)
    """

    def __init__(
        self,
        model_path,
        output_folder,
        ckpt_path,
        script="predict.py",
        python="python",
        use_python=True,
        batch_size=8,
        tmp_folder=None,
    ):
        self.output_folder = Path(output_folder)
        rand = torch.randint(1, 999999999, (1,)).item()
        if tmp_folder is None:
            tmp_folder = self.output_folder
        else:
            tmp_folder = Path(tmp_folder)
        self.eval_path = (tmp_folder / f"eval_{rand}").absolute()
        self.model_path = Path(model_path).absolute()
        script = self.model_path / script
        self.script = script
        self.ckpt_path = Path(ckpt_path).absolute()
        self.batch_size = batch_size
        self.python = python
        self.use_python = use_python

    def evaluate_files(self, file_names, text, file_names_ref=None):
        """Evaluates multiple files

        Arguments
        ---------
        file_names : list
            A list of files

        text : list
            File transcripts (not required for all evaluators)
            Not used in this evaluator

        file_names_ref : list, optional
            A list of reference files / ground truths (if applicable)
            Not used in this evaluator

        Returns
        -------
        result : SpeechEvaluationResult
            a consolidated evaluation result
        """
        current_path = os.getcwd()
        try:
            self.eval_path.mkdir(parents=True, exist_ok=True)
            logger.info("Copying the files to '%s'", self.eval_path)
            for file_name in file_names:
                target_file_name = self.eval_path / Path(file_name).name
                shutil.copy(file_name, target_file_name)

            logger.info("Running evaluation")
            result_path = self.eval_path / "result.txt"
            os.chdir(self.model_path)
            cmd = [
                str(self.script),
                "--mode",
                "predict_dir",
                "--bs",
                str(self.batch_size),
                "--inp_dir",
                str(self.eval_path),
                "--out_path",
                str(result_path),
                "--ckpt_path",
                str(self.ckpt_path),
            ]
            if self.use_python:
                cmd = [self.python] + cmd

            output = subprocess.check_output(cmd)
            logger.info("Evaluation finished, output: %s", output)
            file_names = [path.name for path in self.eval_path.glob("*.wav")]
            with open(result_path) as result_path:
                scores = [float(line.strip()) for line in result_path]
            score_map = dict(zip(file_names, scores))
            scores_ordered = [
                score_map[Path(file_name).name] for file_name in file_names
            ]
            return SpeechEvaluationResult(
                scores_ordered, {"utmos": scores_ordered}
            )
        finally:
            os.chdir(current_path)
            shutil.rmtree(self.eval_path)

class EvaluationBrain:
    """A wrapper similar to the core Brain class that runs standalone evaluation

    Arguments
    ---------
    hparams : dict
        Raw hyperparameters for evaluation

    device : str
        The device on which evaluation will be performed

    """
    def __init__(self, hparams, device="cpu"):
        self.hparams = SimpleNamespace(**hparams)
        self.device = device
        modules = self.hparams.modules
        self.modules = ModuleDict(modules).to(self.device)
        suffix = f"_{self.hparams.eval_suffix}" if self.hparams.eval_suffix else ""
        eval_folder = f"eval_{self.hparams.eval_dataset}{suffix}"
        self.output_folder = Path(self.hparams.output_folder) / eval_folder
        self.samples_folder = self.output_folder / "samples"
        self.samples_folder.mkdir(parents=True, exist_ok=True)
        self.modules.model.vocoder = None
        self.enabled_evaluators = set(self.hparams.evaluations.split(","))
        self.checkpointer = hparams.get("checkpointer", None)
        evaluators = hparams.get("evaluators", {})
        if evaluators:
            self.evaluators = {
                key: evaluator_f(run_opts={"device": device})
                for key, evaluator_f in evaluators.items()
                if key in self.enabled_evaluators
            }
        else:
            self.evaluators = {}

        bulk_evaluators = getattr(self.hparams, "bulk_evaluators", {})
        if bulk_evaluators:
            self.bulk_evaluators = {
                key: evaluator_f()
                for key, evaluator_f in bulk_evaluators.items()
                if key in self.enabled_evaluators
            }
        else:
            self.bulk_evaluators = {}

        if not self.evaluators and not self.bulk_evaluators:
            logger.warn("No evaluators were defined - this run will produce samples only")

        self.attention = []

    def evaluate(self, dataset):
        """Runs evaluation on a dataset

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            a dataset
        """
        if self.checkpointer is not None:
            logger.info("Recovering the checkpoint")
            ckpt = self.hparams.checkpointer.recover_if_possible()
            if not ckpt:
                raise ValueError("Unable to recover the checkpoint")
        self.modules.model.eval()
        self.tracker = Tracker(
            file_name=self.get_tracker_file_name()
        )
        self.on_evaluation_start(dataset)
        if self.hparams.eval_samples is not None:
            dataset = dataset.filtered_sorted(select_n=self.hparams.eval_samples)
        dataset = self.tracker.filter(dataset)
        loader = make_dataloader(dataset, batch_size=self.hparams.batch_size)
        loader_it = iter(loader)
        self.create_reports()
        self.modules.model.show_inference_progress = False
        self.item_ids = self.tracker.get_processed()
        details_keys = list(self.evaluators.keys()) + list(self.bulk_evaluators.keys())
        self.details = {
            evaluator_key: []
            for evaluator_key in details_keys
        }
        self.read_reports()
        self.sample_text = []
        self.sample_file_names = []
        self.ref_file_names = []
        logger.info("Starting evaluation")
        batch_count = math.ceil(len(dataset) / self.hparams.batch_size)
        for batch in tqdm(loader_it, desc="Evaluation", total=batch_count):
            self.evaluate_batch(batch)
        self.evaluate_bulk()
        self.on_evaluation_end(dataset)
        self.write_summary()
        logger.info("Evaluation done")

    def on_evaluation_start(self, dataset):
        """Invoked at the beginning of the evaluation cycle. The default
        implementation is a no-op

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            a dataset
        """
        pass

    def on_evaluation_end(self, dataset):
        """Invoked at the beginning of the evaluation cycle. The default
        implementation is a no-op

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            a dataset
        """
        pass

    def create_reports(self):
        """Creates report files and report writers"""
        self.report_files = {}
        self.report_writers = {}
        for evaluator_key in self.enabled_evaluators:
            columns = self.get_report_columns(evaluator_key)
            file_name = self.output_folder / f"{evaluator_key}.csv"
            resume = file_name.exists() and file_name.stat().st_size > 0
            report_file = open(file_name, "a+")
            self.report_files[evaluator_key] = report_file
            writer = csv.DictWriter(report_file, columns)
            if not resume:
                writer.writeheader()
            self.report_writers[evaluator_key] = writer

    def read_reports(self):
        """Invoked when resuming"""
        for evaluator_key in self.enabled_evaluators:
            file_name = self.output_folder / f"{evaluator_key}.csv"
            if file_name.exists():
                logger.info("%s exists, reading")
                with open(file_name) as report_file:
                    reader = csv.DictReader(report_file)
                    for row in reader:
                        del row["uttid"]
                        row = {key : handle_number(value) for key, value in row.items()}
                        self.details[evaluator_key].append(row)

    def get_tracker_file_name(self):
        """Determines the file name of the tracker file"""
        suffix = f"_{self.hparams.eval_suffix}" if self.hparams.eval_suffix else ""
        file_name = f"tracker_{self.hparams.eval_dataset}{suffix}.txt"
        return self.output_folder / file_name

    def get_report_columns(self, evaluator_key):
        """Returns the columns for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            the identifier of the evaluator

        Returns
        -------
        columns : list[str]
            a list of column headers
        """
        bogus_wavs = torch.randn(2, 10000, device=self.device)
        bogus_length = torch.tensor([1., 1.], device=self.device)
        if evaluator_key in self.evaluators:
            evaluator = self.evaluators[evaluator_key]
            result = evaluator.evaluate(
                wavs=bogus_wavs,
                length=bogus_length,
                text=["BOGUS"] * len(bogus_wavs),
                wavs_ref=bogus_wavs,
                length_ref=bogus_length,
            )
        else:
            bogus_file_name = self.output_folder / "bogus.wav"
            evaluator = self.bulk_evaluators[evaluator_key]
            write_audio(
                str(bogus_file_name),
                bogus_wavs[0].cpu(),
                samplerate=self.hparams.model_sample_rate,
            )
            result = evaluator.evaluate_files(
                file_names=[bogus_file_name],
                text=["BOGUS"],
                file_names_ref=[bogus_file_name],
            )

        return ["uttid"] + list(result.details.keys())

    def evaluate_batch(self, batch):
        """Runs evaluation on a single batch of speech

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            the batch to be evaluated"""
        with torch.no_grad():
            batch = batch.to(self.device)
            wav, length, details = self.create_samples(batch)
            if wav.dim() > 2:
                wav = wav.squeeze(1)

            self.save_samples(batch, wav, length)
            self.item_ids.extend(batch.uttid)
            for evaluator_key, evaluator in self.evaluators.items():
                result = evaluator.evaluate(
                    wavs=wav,
                    length=length,
                    text=batch.label_norm_eval,
                    wavs_ref=batch.sig.data,
                    length_ref=batch.sig.lengths,
                    sample_rate_ref=self.hparams.sample_rate,
                    sample_rate=self.hparams.model_sample_rate
                )
                details = undo_batch(result.details)
                self.write_result(evaluator_key, batch.uttid, details)
                self.details[evaluator_key].extend(details)
            self.tracker.mark_processed(batch.uttid)

    def evaluate_bulk(self):
        """Performs bulk evaluation"""
        for evaluator_key, evaluator in self.bulk_evaluators.items():
            result = evaluator.evaluate_files(
                file_names=self.sample_file_names,
                text=self.sample_text,
                file_names_ref=self.ref_file_names,
            )
            self.details[evaluator_key].append(result.details)
            details = undo_batch(result.details)
            self.write_result(evaluator_key, self.item_ids, details)

    def write_result(self, evaluator_key, uttid, details):
        """Outputs the result details to the report for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            The evaluator key
        batch : list
            The list of IDs
        details : list
            a list of evaluation details, one dictionary per item
        """
        writer = self.report_writers[evaluator_key]
        for uttid, details_item in zip(uttid, details):
            report_details = {
                "uttid": uttid,
                **details_item,
            }
            writer.writerow(
                ascii_only(flatten(report_details))
            )
        self.report_files[evaluator_key].flush()

    def save_samples(self, batch, wav, length):
        """Saves the samples generated by the TTS system

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            the batch being evaluated
        wav : torch.Tensor
            the waveform
        length: torch.Tensor
            relative lengths
        """
        wav_length_abs = (length * wav.size(1)).int()
        for item_id, infer_wav, wav_length in zip(
            batch.uttid, wav, wav_length_abs
        ):
            file_name = str(
                self.samples_folder / f"{item_id}_pred.wav"
            )
            infer_wav_cut = infer_wav[:wav_length.item()].cpu()
            write_audio(
                file_name, infer_wav_cut, samplerate=self.hparams.model_sample_rate
            )
            self.sample_file_names.append(file_name)

    def write_summary(self):
        """Outputs summarized statistics"""
        summary = self.compute_summary()
        file_name = self.output_folder / "summary.json"
        with open(file_name, "w") as output_file:
            json.dump(summary, output_file, indent=4)

    def compute_summary(self):
        """Computes the summarized statistics"""
        return {
            f"{evaluator_key}_{stat_key}": value
            for evaluator_key in self.enabled_evaluators
            if evaluator_key in self.details
            for metric_key in self.hparams.eval_summary[evaluator_key]["descriptive"]
            for stat_key, value in descriptive_statistics(
                items=self.details[evaluator_key],
                key=metric_key,
            ).items()
        }


class Tracker:
    """A tracker that makes it possible to resume evaluation

    Arguments
    ---------
    file_name : str | path-like
        The path to the tracker file"""
    def __init__(self, file_name):
        self.file_name = Path(file_name)

    def mark_processed(self, item_id):
        """Marks the specified file as processed

        Arguments
        ---------
        item_id : str|enumerable
            The item ID or a list of IDS
        """
        if isinstance(item_id, str):
            item_id = [item_id]
        with open(self.file_name, "a+") as tracker_file:
            for item in item_id:
                print(item, file=tracker_file)

    def filter(self, dataset):
        """Filters a dataset using the tracker file

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            A dataset

        Returns
        -------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            The dataset, possibly filtered
        """
        if self.file_name.exists():
            with open(self.file_name) as tracker_file:
                processed_ids = set(
                    line.strip()
                    for line in tracker_file
                )
                remaining_ids = [
                    data_id for data_id in dataset.data_ids
                    if data_id not in processed_ids
                ]
                logger.info(
                    "Tracker %s already exists, %d items already processed, %d items remaining",
                    self.file_name,
                    len(processed_ids),
                    len(remaining_ids)
                )
                dataset = FilteredSortedDynamicItemDataset(dataset, remaining_ids)
        else:
            logger.info("Tracker %s does not exist, evaluating from the beginning")
        return dataset

    def get_processed(self):
        """Retrieves the IDs of items that have been processed

        Returns
        -------
        processed_ids : list
            The list of file IDs
        """
        if self.file_name.exists():
            with open(self.file_name, "r") as tracker_file:
                processed_ids = [
                    line.strip()
                    for line in tracker_file
                ]
        else:
            processed_ids = []
        return processed_ids


RE_INTEGER = re.compile(r"^-?\d+$")
RE_FLOAT = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def handle_number(value):
    """Converts a value to a number, if applicable"""
    if RE_INTEGER.match(value):
        value = int(value)
    elif RE_FLOAT.match(value):
        value = float(value)
    return value


def descriptive_statistics(items, key):
    """Computes descriptive statistics for the summary

    Arguments
    ---------
    items : list
        a list of dictionaries with metric values for each item
    key : str
        The key of the metric for which the statistics will be computed

    Returns
    -------
    statistics : dict
        The desccriptive statistics computed
            <key>_mean : the arithmetic mean
            <key>_std : the standard deviation
            <key>_min : the minimum value
            <key>_max : the maximum value
            <key>_median : the median value
            <key>_q1 : the first quartile
            <key>_q3 : the third quartile
            <key>_iqr : the interquartile ratio
    """
    values = torch.tensor([item[key] for item in items])
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    q1, median, q3 = values.quantile(quantiles)
    stats = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    return {
        f"{key}_{stat_key}": value.item()
        for stat_key, value in stats.items()
    }


RE_NON_ASCII = re.compile(r'[^\x00-\x7F]+')


def ascii_only(values):
    return {
        key: RE_NON_ASCII.sub('', value) if isinstance(value, str)
        else value
        for key, value in values.items()
    }


def flatten(value):
    """Converts tensors to scalars and lists of strings to strings

    Arguments
    ---------
    value : dict
        the dictionary to flatten

    Returns
    -------
    result : dict
        a flattened dictionary
    """
    return {
        key: item_value.item() if torch.is_tensor(item_value) else item_value
        for key, item_value in value.items()
    }
