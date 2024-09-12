import speechbrain as sb
import re
import string
import sys
import logging
import torch
import torchaudio

from functools import partial
from hyperpyyaml import load_hyperpyyaml
from speechbrain.inference.eval import EvaluationBrain
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.utils.distributed import run_on_main
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSEvaluationBrain(EvaluationBrain):
    """A brain implementation for the evaluation of
    external (i.e. non-SpeechBrain) TTS systems"""

    def on_evaluation_start(self, dataset):
        if self.hparams.spk == "random":
            self.spk = self.select_random_spk()
        else:
            self.spk = self.hparams.spk
        self.modules.model.to(self.device)

    def select_random_spk(self):
        """Selects a random item from the dataset
        for a speaker prompt

        Returns
        -------
        wav : torch.Tensor
            The waveform
        text : str
            The transcription of the waveform
        """
        generator = torch.Generator()
        generator.manual_seed(self.hparams.seed)
        data_idx = torch.randint(0, len(dataset), (1,), generator=generator).item()
        data = dataset[data_idx]
        logger.info("Using '%s' for the audio prompt", data["uttid"])
        wav = torchaudio.functional.resample(
            data["sig"],
            orig_freq=self.hparams.sample_rate,
            new_freq=self.hparams.model_sample_rate,
        ).to(self.device)
        return (wav, data["label_norm"])

    def create_samples(self, batch):
        batch = batch.to(self.device)
        spk = (
            (batch.sig_random_match, batch.text_random_match)
            if self.hparams.spk == "random_match"
            else self.spk
        )
        result = self.modules.model(
            text=batch.label_norm,
            spk=spk,
            **self.hparams.model_args
        )
        details = {"tokens": result.tokens}
        return result.wav, result.length, details


RE_PUNCTUATION = re.compile(
    "|".join(
        re.escape(char) for char in string.punctuation
    )
)


@sb.utils.data_pipeline.takes("label")
@sb.utils.data_pipeline.provides(
    "label_norm", "label_norm_length", "label_norm_eval"
)
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
    label_norm = label.upper()
    yield label_norm
    yield len(label_norm)
    label_norm_eval = RE_PUNCTUATION.sub("", label_norm)
    yield label_norm_eval


@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_ref_pipeline(wav):
    """The audio loading pipeline for references

    Arguments
    ---------
    wav : str
        The file path

    Returns
    -------
    sig : torch.Tensor
        The waveform
    """
    sig = sb.dataio.dataio.read_audio(wav)

    return sig


def dataio_prepare(hparams):
    """Prepares the dataset

    Arguments
    ---------
    hparams : dict
        Raw hyperparameters"""

    data_folder = hparams["data_folder"]
    eval_dataset = hparams["eval_dataset"]
    json_path = hparams[f"{eval_dataset}_json"]

    dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=json_path,
        replacements={"data_root": data_folder},
        output_keys=["uttid", "label"],
    )
    dataset = select_subset(dataset, hparams)
    dataset.add_dynamic_item(label_norm_pipeline)
    dataset.add_dynamic_item(audio_ref_pipeline)

    output_keys = [
        "uttid", "label_norm_eval", "label_norm", "label_norm_length", "sig"
    ]

    if hparams["spk"] == "random_match":
        spk_idx, spk_samplers = group_by_speaker(
            dataset,
            hparams
        )
        spk_sample = {}

        def spk_random_match(uttid, dataset, spk_sample):
            # Sample a speaker-matched embedding
            selected_idx = spk_sample[uttid]

            # Retrieve the embedding value from the dataset
            with dataset.output_keys_as(["sig", "label_norm_eval"]):
                sig = dataset[selected_idx]["sig"]
                text = dataset[selected_idx]["label_norm_eval"]
            yield sig
            yield text

        spk_emb_random_match_pipeline = partial(
            spk_random_match,
            spk_sample=spk_sample,
            dataset=dataset.filtered_sorted(),
        )
        resample_fn = partial(
            resample_spk,
            spk_idx=spk_idx,
            sample=spk_sample,
            dataset=dataset,
            spk_samplers=spk_samplers
        )
        resample_fn(epoch=0)

        dataset.add_dynamic_item(
            func=spk_emb_random_match_pipeline,
            takes=["uttid"],
            provides=["sig_random_match", "text_random_match"],
        )
        output_keys = output_keys + ["sig_random_match", "text_random_match"]

    dataset.set_output_keys(output_keys)

    if hparams["sorting"] == "ascending":
        dataset = dataset.filtered_sorted(sort_key="label_norm_length")
    elif hparams["sorting"] == "descending":
        dataset = dataset.filtered_sorted(
            sort_key="label_norm_length", reverse=True
        )
    return dataset


def select_subset(dataset, hparams):
    """Selects a subset of the dataset provided, if specified.
    The selection is controlled by a hyperparameter named
    eval_subset, which is expected to list the IDs of the
    data items on which evaluation will take place, one per line

    Arguments
    ---------
    dataset : speechbrain.dataio.dataset.DynamicItemDataset
        A dataset
    hparams : dict
        A hyperparameters file

    Returns
    -------
    subset : dataset
        The dataset, filtered down if applicable
    """
    eval_subset_path = hparams.get("eval_subset")
    if eval_subset_path is not None:
        eval_subset_path = Path(eval_subset_path)
        if not eval_subset_path.exists():
            raise ValueError(f"eval_subset {eval_subset_path} does not exist")
        with open(eval_subset_path) as eval_subset_file:
            eval_subset_ids = [line.strip() for line in eval_subset_file]
        subset = FilteredSortedDynamicItemDataset(dataset, eval_subset_ids)
    else:
        subset = dataset
    return subset


def group_by_speaker(dataset, hparams):
    """Groups utterance IDs in a dataset by speaker, for selection. The selection
    is stable based on the seed - calling this method multiple times will always
    result in the same order

    Arguments
    ---------
    dataset : torch.Tensor
        the dataset from which to select items
    hparams : dict
        hyperparameters
    
    Returns
    -------
    spk_idx : dict
        a str -> int dictionary with a list of utterance indexes
        for every speaker
    spk_samplers : dict
        a reproducible sampler for every speaker
    spk_samplers_it : dict
        an iterator for each sampler
    """
    spk_idx = {}
    spk_samplers = {}
    speakers = []
    generator = torch.Generator()
    generator.manual_seed(hparams["seed"])
    min_length = hparams.get("spk_match_min_length")

    # Group by speaker
    longest = {}
    longest_idx = {}
    spk_set = set([])
    with dataset.output_keys_as(["spk_id", "label"]):
        for idx, item in enumerate(dataset):
            spk_id = item["spk_id"]
            length = len(item["label"])
            spk_set.add(spk_id)
            if length > longest.get(spk_id, 0):
                longest[spk_id] = length
                longest_idx[spk_id] = idx
            if min_length is not None and len(item["label"]) < min_length:
                continue
            if spk_id not in spk_idx:
                spk_idx[spk_id] = []
                speakers.append(spk_id)
            spk_idx[spk_id].append(idx)

    missing = spk_set - set(spk_idx.keys())
    for spk_id in missing:
        spk_idx[spk_id] = [longest_idx[spk_id]]
        speakers.append(spk_id)

    # Create a reproducible sampler
    for spk_id in speakers:
        sampler = hparams["spk_sampler"](data_source=spk_idx[spk_id])
        spk_samplers[spk_id] = sampler

    return spk_idx, spk_samplers


def looping_iterator(items):
    while True:
        for item in iter(items):
            yield item


def resample_spk(sample, spk_idx, spk_samplers, dataset, epoch):
    """Selects new samples

    Arguments
    ---------
    spk_idx : dict
        Data item indexes grouped by speaker
    spk_samplers : dict
        A sampler for each speaker
    spk_samplers_it : dict
        An iterator for each speaker
    epoch : int
        The epoch number

    Returns
    -------
    sample : dict
        a dictionary with uttids as keys and matching
        indexes as values
    """
    if epoch is None:
        epoch = 0
    spk_samplers_it = {}
    for spk_id, sampler in spk_samplers.items():
        sampler.set_epoch(epoch)
        spk_samplers_it[spk_id] = looping_iterator(sampler)
    with dataset.output_keys_as(["uttid", "spk_id"]):
        for item in dataset:
            spk_item_idx = next(spk_samplers_it[item["spk_id"]])
            dataset_item_idx = spk_idx[item["spk_id"]][spk_item_idx]
            sample[item["uttid"]] = dataset_item_idx


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(
            fin, overrides, overrides_must_match=False
        )

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

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from libritts_prepare import prepare_libritts
    # Data preparation, to be run on only one process.    
    if not hparams["skip_prep"]:
        eval_dataset = hparams["eval_dataset"]
        run_on_main(
            prepare_libritts,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["prepare_save_folder"],
                "save_json_train": hparams["train_json"],
                "save_json_valid": hparams["valid_json"],
                "save_json_test": hparams["test_json"],
                "sample_rate": hparams["sample_rate"],
                "train_split": hparams["train_split"] if eval_dataset == "train" else None,
                "valid_split": hparams["valid_split"] if eval_dataset == "valid" else None,
                "test_split": hparams["test_split"] if eval_dataset == "test" else None,
                "seed": hparams["seed"],
                "model_name": hparams["model"].__class__.__name__,
                "min_utterance_length": None,
                "device": run_opts.get("device", "cpu"),
            },
        )

    dataset = dataio_prepare(hparams)

    brain = TTSEvaluationBrain(hparams, device=run_opts.get("device", "cpu"))
    brain.evaluate(dataset)
