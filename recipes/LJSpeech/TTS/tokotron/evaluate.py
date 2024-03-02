"""Evaluates a checkpoint using an MOS estimation tool

Authors
* Artem Ploujnikov 2024
"""

import speechbrain as sb
import json
import logging
import math
import sys
import csv
import torch
import string
import re
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from types import SimpleNamespace
from torch.nn import ModuleDict
from tqdm.auto import tqdm
from speechbrain.dataio.batch import undo_batch
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


class TokotronEvaluator:
    """An evaluator class for the TTS model
    
    Arguments
    ---------
    hparams: dict
        hyperparameters (as a dictionary)
    device : str | torch.device
        the device
    """
    def __init__(self, hparams, device):
        self.hparams = SimpleNamespace(**hparams)
        self.device = device
        modules = self.hparams.modules
        self.modules = ModuleDict(modules).to(self.device)
        self.output_folder = Path(self.hparams.output_folder) / "eval"
        self.samples_folder = self.output_folder / "samples"
        self.samples_folder.mkdir(parents=True, exist_ok=True)
        evaluators = hparams.get("evaluators", {})
        if evaluators:
            self.evaluators = {
                key: evaluator_f(run_opts={"device": device})
                for key, evaluator_f in evaluators.items()
            }
        else:
            logger.warn("No evaluators were defined - this run will produce samples only")
            self.evaluators = {}

    def evaluate(self, dataset):
        """Runs evaluation on a dataset

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            a dataset
        """
        logger.info("Recovering the checkpoint")
        ckpt = self.hparams.checkpointer.recover_if_possible()
        if not ckpt:
            raise ValueError("Unable to recover the checkpoint")
        loader = sb.dataio.dataloader.make_dataloader(dataset, batch_size=self.hparams.batch_size)
        loader_it = iter(loader)
        self.create_reports()
        self.modules.model.show_inference_progress = False
        self.item_ids = []
        self.details = {
            evaluator_key: []
            for evaluator_key in self.evaluators
        }
        logger.info("Starting evaluation")
        batch_count = math.ceil(len(dataset) / self.hparams.batch_size)
        for batch in tqdm(loader_it, desc="Evaluation", total=batch_count):
            self.evaluate_batch(batch)
        self.write_summary()
        logger.info("Evaluation done")

    def create_reports(self):
        """Creates report files and report writers"""
        self.report_files = {}
        self.report_writers = {}
        for evaluator_key in self.evaluators:
            columns = self.get_report_columns(evaluator_key)
            file_name = self.output_folder / f"{evaluator_key}.csv"
            report_file = open(file_name, "w")
            self.report_files[evaluator_key] = report_file
            writer = csv.DictWriter(report_file, columns)
            writer.writeheader()
            self.report_writers[evaluator_key] = writer

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
        evaluator = self.evaluators[evaluator_key]
        result = evaluator.evaluate(
            wavs=bogus_wavs,
            length=bogus_length,
            text="BOGUS",
        )
        return ["uttid"] + list(result.details.keys())

    def evaluate_batch(self, batch):
        """Runs evaluation on a single batch of speech

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            the batch to be evaluated"""
        batch = batch.to(self.device)
        tokens, tokens_length = batch.tokens
        self.modules.model.vocoder.device = self.device
        infer_out = self.modules.model.infer(
            input_tokens=tokens, input_length=tokens_length
        )
        self.save_samples(batch, infer_out)
        for evaluator_key, evaluator in self.evaluators.items():
            result = evaluator.evaluate(
                wavs=infer_out.wav,
                length=infer_out.wav_length,
                text=batch.label_norm_eval
            )
            details = undo_batch(result.details)
            self.write_result(evaluator_key, batch, details)
            self.details[evaluator_key].extend(details)

    def write_result(self, evaluator_key, batch, details):
        """Outputs the result details to the report for the specified evaluator
        
        Arguments
        ---------
        evaluator_key : str
            The evaluator key
        batch : speechbrain.dataio.batch.PaddedBatch
            The batch evaluation
        details : list
            a list of evaluation details, one dictionary per item
        """
        writer = self.report_writers[evaluator_key]
        for uttid, details_item in zip(batch.uttid, details):
            report_details = {
                "uttid": uttid,
                **details_item,
            }
            writer.writerow(flatten(report_details))
        self.report_files[evaluator_key].flush()

    def save_samples(self, batch, infer_out):
        """Saves the samples generated by the TTS system

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            the batch being evaluated
        infer_out : speechbrain.lobes.models.discrete.Tokotron.TokotronInfernceOutput
            the model output
        """
        wav_length_abs = (infer_out.wav_length * infer_out.wav.size(1)).int()
        for item_id, infer_wav, wav_length in zip(
            batch.uttid, infer_out.wav, wav_length_abs
        ):
            file_name = str(
                self.samples_folder / f"{item_id}_pred.wav"
            )
            infer_wav_cut = infer_wav[:wav_length.item()].cpu()
            sb.dataio.dataio.write_audio(
                file_name, infer_wav_cut, samplerate=self.hparams.model_sample_rate
            )

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
            for evaluator_key in self.evaluators
            for metric_key in self.hparams.eval_summary[evaluator_key]["descriptive"]
            for stat_key, value in descriptive_statistics(
                items=self.details[evaluator_key],
                key=metric_key,
            ).items()
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


RE_PUNCTUATION = re.compile(
    "|".join(
        re.escape(char) for char in string.punctuation
    )
)


@sb.utils.data_pipeline.takes("label_norm")
@sb.utils.data_pipeline.provides("label_norm_eval")
def label_norm_pipeline(label):
    """Normalizes labels for ASR comparison, converting to uppercase and removing
    punctuation

    Arguments
    ---------
    label : str
        The unnormalized label

    Returns
    -------
    result : str
        The normalized label
    """
    label = label.upper()
    label = RE_PUNCTUATION.sub("", label)
    return label


def descriptive_statistics(items, key):
    """Computes descriptive statistics for the summary
    
    Arguments
    ---------
    items : list
        a list of dictionaries with metric values for each item
    key : str
        """
    values = torch.tensor([item[key] for item in items])
    stats = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
    }
    return {
        f"{key}_{stat_key}": value.item()
        for stat_key, value in stats.items()
    }


if __name__ == "__main__":
    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Reuse the preparation function from the training script
    from train import dataio_prepare

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides, overrides_must_match=False)

    # Load evaluation hyperparameters
    eval_hparams_file = hparams.get("eval_hparams")
    if eval_hparams_file is None:
        # If not defined, look for eval.yaml in the same folder
        # as the original hyperparameters file
        eval_hparams_file = Path(hparams_file).parent / "eval.yaml"
    if eval_hparams_file.exists():
        logger.info(
            "Using evaluation hyperparameters from %s",
            eval_hparams_file
        )
        eval_hparams = load_hyperpyyaml(
            eval_hparams_file, overrides, overrides_must_match=False
        )
        hparams.update(eval_hparams)
    else:
        logger.info(
            "%s not found - not using evaluation hyperparameters",
            eval_hparams_file
        )

    from ljspeech_prepare import prepare_ljspeech

    # Data Preparation
    if not hparams["skip_prep"]:
        with hparams["freezer"]:
            run_on_main(
                prepare_ljspeech,
                kwargs={
                    "data_folder": hparams["data_folder"],
                    "save_folder": hparams["prepare_save_folder"],
                    "splits": hparams["splits"],
                    "split_ratio": hparams["split_ratio"],
                    "seed": hparams["seed"],
                    "extract_features": ["audio_tokens"],
                    "extract_features_opts": hparams["extract_features_opts"],
                    "extract_phonemes": hparams["input"] == "phonemes",
                    "model_name": "tokotron",
                    "g2p_src": hparams["g2p_src"],
                    "skip_ignore_folders": hparams["prepare_skip_ignore_folders"],
                    "frozen_split_path": hparams.get("frozen_split_path"),
                    "device": run_opts.get("device", "cpu"),
                },
            )

    # Reading command line arguments
    datasets, _ = dataio_prepare(hparams)

    # Select the dataset to use in evaluation
    eval_dataset_key = hparams.get("eval_dataset", "valid")
    eval_dataset = datasets[eval_dataset_key]
    eval_dataset.add_dynamic_item(
        label_norm_pipeline
    )
    eval_dataset.set_output_keys(
        ["uttid", "label_norm_eval", "tokens"]
    )

    # Create the evaluator
    eval = TokotronEvaluator(hparams, device=run_opts["device"])

    # Start evaluation
    logger.info("Evaluating on %s", eval_dataset_key)
    eval.evaluate(eval_dataset)
