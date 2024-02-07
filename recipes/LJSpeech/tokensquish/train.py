#!/usr/bin/env/python3
"""Recipe for training a token compression model

Authors
 * Artem Ploujnikov 2024
"""


import logging
import speechbrain as sb
import torch
import sys
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.preparation import add_prepared_features
from speechbrain.utils.audio_tokens import (
    get_silence_token,
    use_silence_padding,
    feature_pad_to,
)
from speechbrain.dataio.dataio import length_to_mask

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_COUNT = 3


# Brain class for speech recognition training
class TokenSquishBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs all the computation of the Tokotron TTS

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            TTS predictions
        """
        batch = batch.to(self.device)
        audio_tokens, audio_tokens_length = batch.audio_tokens_pad
        audio_tokens = self.modules.model.compress(audio_tokens)
        predictions = self.modules.model(
            tokens=audio_tokens, length=audio_tokens_length,
        )

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)
        audio_tokens, lengths = batch.audio_tokens_pad
        batch_size, out_len, heads, tok_dim = predictions.p_seq.shape
        lengths_reshaped = lengths.repeat(heads)
        p_seq_reshaped = (
            predictions.p_seq.transpose(1, 2).reshape(
                batch_size * heads, out_len, tok_dim
            )
        )[:, :out_len, :]
        batch_size, audio_len, heads = audio_tokens.shape
        audio_tokens_reshaped = audio_tokens.transpose(1, 2).reshape(
            batch_size * heads, audio_len
        )
        seq_loss = self.hparams.compute_cost(
            p_seq_reshaped, audio_tokens_reshaped, length=lengths_reshaped,
        )
        lengths_abs = lengths * audio_tokens.size(1)
        enc_alignment = get_alignment(predictions.enc_attn)
        enc_attn_loss = self.hparams.compute_cost_attn(
            enc_alignment,
            input_lengths=lengths_abs,
            target_lengths=lengths_abs,
            max_input_len=enc_alignment.size(2),
            max_target_len=enc_alignment.size(1),
        )
        dec_alignment = get_alignment(predictions.dec_attn)
        dec_attn_loss = self.hparams.compute_cost_attn(
            dec_alignment,
            input_lengths=lengths_abs,
            target_lengths=lengths_abs,
            max_input_len=dec_alignment.size(2),
            max_target_len=dec_alignment.size(1),
        )

        attn_loss = 0.5 * (enc_attn_loss + dec_attn_loss)
        loss = seq_loss + self.hparams.guided_attention_weight * attn_loss
        self.acc_metric.append(
            batch.uttid, predictions.p_seq, audio_tokens, lengths
        )

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        self.acc_metric = sb.utils.metric_stats.MetricStats(accuracy)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        acc_stats = self.acc_metric.summarize()
        stage_stats = {
            "loss": stage_loss,
            **{f"acc_{key}": value for key, value in acc_stats.items()},
        }
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            if self.hparams.lr_annealing_mode == "epoch":
                _, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            lr = self.optimizer.param_groups[0]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, max_keys=["acc_average"],
            )

    def fit_batch(self, batch):
        loss = super().fit_batch(batch)
        if self.hparams.lr_annealing_mode == "step":
            self.hparams.lr_annealing(self.optimizer)
        return loss


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    silence_token : dict
        the token used for silence
    """

    # Define datasets from json data manifest file
    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }

    silence_token, _ = get_silence_token(hparams["token_model"])
    silence_token = silence_token.cpu()
    audio_bos = (
        torch.ones(1, hparams["audio_tokens_per_step"]) * hparams["bos_index"]
    )

    @sb.utils.data_pipeline.takes("audio_tokens")
    @sb.utils.data_pipeline.provides("audio_tokens_pad")
    def audio_pipeline(audio_tokens):
        audio_tokens = torch.from_numpy(audio_tokens)
        audio_tokens_pad = feature_pad_to(
            audio_tokens, len(audio_tokens), silence_token
        )
        yield audio_tokens_pad
        audio_tokens_bos = torch.cat([audio_bos, audio_tokens_pad], dim=0)
        yield audio_tokens_bos

    for dataset in data_info:
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline],
            output_keys=["uttid", "audio_tokens_pad"],
        )

        add_prepared_features(
            dataset=dynamic_dataset,
            save_path=Path(hparams["prepare_save_folder"]) / "features",
            id_key="uttid",
            features=["audio_tokens"],
        )

        datasets[dataset] = dynamic_dataset
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="length")
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    return datasets, silence_token


def get_alignment(attn):
    while attn.dim() > 3:
        attn = attn.mean(1)
    return attn


def read_token_list(file_name):
    """Reads a simple text file with tokens (e.g. characters or phonemes) listed
    one per line

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    result: list
        a list of tokens
    """
    if not Path(file_name).exists():
        raise ValueError(f"Token file {file_name} not found")
    with open(file_name) as token_file:
        return [line.strip("\r\n") for line in token_file if line]


def apply_overfit_test(hparams, dataset):
    """Helper for applying an overfit test conditionally based
    on hyperparameters:

    `overfit_test`: whether or not to apply an overfit test
    `overfit_test_sample_count`: the number of samples to use from the
        original dataset
    `overfit_test_epoch_data_count`: the number of samples per epoch

    The function will accept datasets, (train, valid, test) tuples
    or dictionaries of the form:
    {"train": dataset1, "valid": dataset2, "test": dataset3}

    If a tuple or dictionary is used, the training dataset will be of length
    overfit_test_epoch_data_count wheres the evaluation dataset will be of
    length overfit_test_sample_count.

    Arguments
    ---------
    hparams: dict
        parsed hyperparameters
    dataset: DynamicItemDataset|tuple|dict
        One of the following
        a dataset
        a dictionary ({"train": dataset1, "valid": dataset2, "test": dataset3})
        a (train, valid, test)  tuple of datasets

    Returns
    -------
    result: DynamicItemDataset|tuple|dict
        a dataset or collection of datasets suitable for
        an overfitting test - in the same format as the
        dataset argument (single dataset, dictionary and tuple)
    """
    if hparams["overfit_test"]:
        if isinstance(dataset, tuple):
            dataset_train, _, _ = dataset
            dataset_train = apply_overfit_test(hparams, dataset_train)
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            result = dataset_train, dataset_eval, dataset_eval
        elif isinstance(dataset, dict):
            dataset_train = apply_overfit_test(hparams, dataset["train"])
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            result = {
                "train": dataset_train,
                "valid": dataset_eval,
                "test": dataset_eval,
                "sample": dataset_eval,
            }
        else:
            result = dataset.overfit_test(
                hparams["overfit_test_sample_count"],
                hparams["overfit_test_epoch_data_count"],
            )
    else:
        result = dataset
    return result


def accuracy(p_seq, audio_tokens, length):
    """A simple accuracy metric (matching tokens / length).
    Please note that this metric is not a direct/accurate
    measure of subjective sound quality.

    Arguments
    ---------
    p_seq : torch.Tensor
        The token probabilities
    audio_tokens : torch.Tensor
        Ground truth audio tokens
    length : torch.Tensor
        a 1-D tensor of realtive lengths

    Returns
    -------
    result : float
        The accuracy metric value
    """
    audio_tokens_pred = p_seq.argmax(-1)
    max_len = min(audio_tokens.size(1), audio_tokens_pred.size(1))
    length_abs = length * max_len
    mask = length_to_mask(length_abs, max_len).bool()
    correct = audio_tokens[:, :max_len] == audio_tokens_pred[:, :max_len]
    correct *= mask.unsqueeze(-1)
    correct = correct.sum(-1).sum(-1).float()
    return correct / (length_abs * audio_tokens.size(-1))


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from ljspeech_prepare import prepare_ljspeech

    # Data preparation, to be run on only one process.
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
                    "model_name": "tokotron",
                    "device": run_opts.get("device", "cpu"),
                },
            )

    # We can now directly create the datasets for training, valid, and test
    datasets, silence_token = dataio_prepare(hparams)

    # Apply overfit test settings
    datasets = apply_overfit_test(hparams, datasets)
    token_keys = ["audio_tokens_pad", "audio_tokens_bos"]

    # Trainer initialization
    ts_brain = TokenSquishBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    ts_brain.fit(
        ts_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=use_silence_padding(
            hparams["train_dataloader_opts"], silence_token, token_keys
        ),
        valid_loader_kwargs=use_silence_padding(
            hparams["valid_dataloader_opts"], silence_token, token_keys
        ),
    )

    # Load best checkpoint for evaluation
    test_stats = ts_brain.evaluate(
        test_set=datasets["test"],
        max_key="acc_average",
        test_loader_kwargs=use_silence_padding(
            hparams["test_dataloader_opts"], silence_token, token_keys
        ),
    )

    # Save final checkpoint (fixed name)
    ts_brain.checkpointer.save_checkpoint(name="latest")
