import speechbrain as sb
import re
import string
import sys
import logging
import torch
import torchaudio

from hyperpyyaml import load_hyperpyyaml
from speechbrain.inference.eval import EvaluationBrain
from speechbrain.utils.distributed import run_on_main
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSEvaluationBrain(EvaluationBrain):
    """A brain implementation for the evaluation of
    external (i.e. non-SpeechBrain) TTS systems"""

    def on_evaluation_start(self, dataset):
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
        self.spk = (wav, data["label_norm"])
        self.modules.model.to(self.device)

    def create_samples(self, batch):
        batch = batch.to(self.device)
        result = self.modules.model(
            text=batch.label_norm,
            spk=self.spk,
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
    dataset.add_dynamic_item(label_norm_pipeline)
    dataset.add_dynamic_item(audio_ref_pipeline)
    dataset.set_output_keys(
        ["uttid", "label_norm_eval", "label_norm", "label_norm_length", "sig"]
    )

    if hparams["sorting"] == "ascending":
        dataset = dataset.filtered_sorted(sort_key="label_norm_length")
    elif hparams["sorting"] == "descending":
        dataset = dataset.filtered_sorted(
            sort_key="label_norm_length", reverse=True
        )
    return dataset


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

    from ljspeech_prepare import prepare_ljspeech
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_ljspeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["prepare_save_folder"],
                "splits": hparams["splits"],
                "split_ratio": hparams["split_ratio"],
                "seed": hparams["seed"],
                "model_name": "tokotron",
                "skip_ignore_folders": hparams["prepare_skip_ignore_folders"],
                "frozen_split_path": hparams.get("frozen_split_path"),
                "device": run_opts.get("device", "cpu"),
            },
        )

    dataset = dataio_prepare(hparams)

    brain = TTSEvaluationBrain(hparams, device=run_opts.get("device", "cpu"))
    brain.evaluate(dataset)
