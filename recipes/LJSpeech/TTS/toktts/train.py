#!/usr/bin/env/python3
"""Recipe for training a Text-to-Speech system based on tokenized audio

Inspired by WhisperSpeech
https://github.com/collabora/WhisperSpeech


Authors
 * Artem Ploujnikov
"""

import sys
import torch
import logging
import speechbrain as sb
import multiprocessing
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from tqdm.auto import tqdm
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.preparation import add_prepared_features

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class TokTTSBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs all the computation of the CTC + seq2seq ASR. It returns the
        posterior probabilities of the CTC and seq2seq networks.

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
        tokens, tokens_length = batch.tokens
        audio_tokens, audio_tokens_length = batch.audio_tokens_bos

        out, decoder_multihead_attns = self.compute_tts(
            tokens,
            tokens_length,
            audio_tokens,
            audio_tokens_length,
        )

        p_seq = self.hparams.log_softmax(out)

        predictions = p_seq, decoder_multihead_attns

        return predictions

    def compute_tts(self, tokens, tokens_length, audio_tokens, audio_tokens_length):
        text_emb = self.modules.text_emb(tokens)
        audio_emb = self.modules.audio_emb(audio_tokens)
        batch_size, audio_max_len, heads, audio_dim = audio_emb.shape
        audio_emb_combined = audio_emb.reshape(
            batch_size,
            audio_max_len,
            heads * audio_dim
        )
        tgt = self.modules.tgt_in_proj(audio_emb_combined)
        (
            encoder_out,
            decoder_out,
            encder_self_attns,
            decoder_self_attns,
            decoder_multihead_attns,
        ) = self.modules.model(
            src=text_emb,
            tgt=tgt,
            src_length=tokens_length,
            tgt_length=audio_tokens_length,
        )

        lin_out = self.modules.seq_lin(decoder_out)
        batch_size, text_max_len, _ = lin_out.shape
        lin_out_heads = lin_out.reshape(
            batch_size,
            text_max_len,
            self.hparams.num_quantizers,
            self.hparams.audio_num_tokens
        )
        return lin_out_heads, decoder_multihead_attns

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
        p_seq, decoder_multihead_attns = predictions
        batch = batch.to(self.device)

        audio_tokens, lengths = batch.audio_tokens
        batch_size, out_len, heads, tok_dim = p_seq.shape
        max_len = out_len - 1
        p_seq_reshaped = (
            p_seq
            .transpose(1, 2)
            .reshape(batch_size * heads, out_len, tok_dim)
        )[:, :max_len, :]
        audio_tokens_reshaped = (
            audio_tokens
            .transpose(1, 2)
            .reshape(batch_size * heads, max_len)
        )
        lengths_reshaped = lengths.repeat(heads)
        seq_loss = self.hparams.seq_cost(
            p_seq_reshaped,
            audio_tokens_reshaped,
            length=lengths_reshaped,
            reduction="mean"
        )
        alignments = torch.cat(
            [item.unsqueeze(-1) for item in decoder_multihead_attns],
            dim=-1
        ).mean(dim=-1)
        attn_loss = self.hparams.attn_cost(
            alignments,
            input_lengths=batch.tokens.lengths * batch.tokens.data.size(1),
            target_lengths=lengths * out_len
        )
        loss = seq_loss + self.hparams.guided_attention_weight * attn_loss
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
        # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        self.create_perfect_samples()

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
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            lr = self.optimizer.param_groups[0]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )

            self.create_samples()

    def fit_batch(self, batch):
        loss = super().fit_batch(batch)
        self.hparams.noam_annealing(self.optimizer)
        return loss

    def create_samples(self):
        """Writes audio samples at the end of an epoch"""
        epoch = self.hparams.epoch_counter.current
        samples_folder = Path(self.hparams.samples_folder) / str(epoch)
        samples_folder.mkdir(parents=True, exist_ok=True)
        sample_loader = sb.dataio.dataloader.make_dataloader(
            self.sample_data, **self.hparams.sample_dataloader_opts
        )
        for batch in sample_loader:
            sample_tokens, length, attn = self.infer(batch)
            samples = self.modules.vocoder(sample_tokens, length)
            self.write_samples(
                samples_folder,
                batch.uttid,
                samples,
                length
            )

    def create_perfect_samples(self):
        """Creates the best samples that can be created using
        the vocoder provided, for comparison purposes"""
        samples_folder = Path(self.hparams.samples_folder) / "_perfect"
        if samples_folder.exists():
            return
        samples_folder.mkdir(parents=True, exist_ok=True)
        sample_loader = sb.dataio.dataloader.make_dataloader(
            self.sample_data, **self.hparams.sample_dataloader_opts
        )
        for batch in sample_loader:
            sample_tokens, length = batch.audio_tokens
            samples = self.modules.vocoder(sample_tokens, length)
            self.write_samples(
                samples_folder,
                batch.uttid,
                samples,
                length
            )

    def write_samples(self, samples_folder, item_ids, sig, length):
        """Writes a series of audio samples

        Arguments
        ---------
        samples_folder : path-like
            the destination folder
        item_ids : enumerable
            a list of IDs
        sig : torch.Tensor
            raw waveform
        length: torch.Tensor
            relative lengths
        """
        max_len = sig.size(1)
        for item_id, item_sig, item_length in zip(item_ids, sig, length):
            file_name = samples_folder / f"{item_id}.wav"
            item_length_abs = (item_length * max_len).int().item()
            item_sig_cut = item_sig[:item_length_abs]
            sb.dataio.dataio.write_audio(
                file_name,
                item_sig_cut,
                samplerate=self.hparams.model_sample_rate)

    def infer(self, batch):
        """Performs autorgressive inference

        Arguments
        ---------
        batch: PaddedBatch
            a batch from the sample dataset

        Returns
        -------
        """
        tokens, tokens_length = batch.tokens
        batch_size = len(tokens)
        bos = torch.ones(batch_size, 1, self.hparams.num_quantizers, device=tokens.device) * self.hparams.bos_index
        audio_tokens = bos
        audio_tokens_length = torch.ones(batch_size, device=tokens.device)

        max_tokens = (
            self.hparams.debug_infer_max_audio_tokens
            if self.debug
            else self.hparams.infer_max_audio_tokens
        )
        for _ in tqdm(range(max_tokens), desc="Inference"):
            out, decoder_multihead_attn = self.compute_tts(
                tokens,
                tokens_length,
                audio_tokens,
                audio_tokens_length,
            )
            audio_tokens_out = out.argmax(-1)
            audio_tokens = torch.cat(
                [bos, audio_tokens_out],
                dim=1
            )

        # TODO: Implement gating
        lengths = torch.ones(batch_size)

        return audio_tokens_out, lengths, decoder_multihead_attn


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
    """

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    label_encoder = hparams["label_encoder"]

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides(
        "label", "tokens", "tokens_bos"
    )
    def text_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        label = label.upper()
        yield label
        tokens = label_encoder.encode_label_torch(label)
        yield tokens

    audio_bos = torch.ones(1, hparams["num_quantizers"]) * hparams["bos_index"]

    @sb.utils.data_pipeline.takes("audio_tokens")
    @sb.utils.data_pipeline.provides(
        "audio_tokens_bos"
    )
    def audio_add_bos(audio_tokens):
        return torch.cat(
            [
                audio_bos,
                torch.from_numpy(audio_tokens)
            ],
            dim=0
        )

    init_sequence_encoder(hparams)

    for dataset in data_info:
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=[text_pipeline, audio_add_bos],
            output_keys=[
                "uttid",
                "tokens",
                "audio_tokens",
                "audio_tokens_bos",
            ],
        )

        add_prepared_features(
            dataset=dynamic_dataset,
            save_path=Path(hparams["save_folder"]) / "features",
            id_key="uttid",
            features=["audio_tokens"]
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

    datasets["sample"] = datasets["valid"].batch_shuffle(1).filtered_sorted(
        select_n=hparams["num_audio_samples"]
    )
    return datasets


def init_sequence_encoder(hparams):
    """Initialize a sequence encoder

    Arguments
    ---------
    hparams: dict
        parsed hyperparameters
    prefix: str
        the prefix to be prepended to hyperparameter keys, per the naming
        convention

        {prefix}_label_encoder: the hparams key for the label encoder
        {prefix}_list_file:  the hparams key for the list file

    Returns
    -------
    encoder: speechbrain.dataio.encoder.TextEncoder
        an encoder instance"""
    encoder = hparams["label_encoder"]
    token_list_file_name = hparams["token_list_file"]
    tokens = read_token_list(token_list_file_name)
    encoder.add_unk()
    encoder.add_bos_eos()
    encoder.update_from_iterable(tokens, sequence_input=False)
    return encoder


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


def check_multiprocessing(hparams, run_opts=None):
    """Adjusts multiprocessing parameters depending on the operating
    system's capabilities"""
    if run_opts is None:
        run_opts = {}
    start_methods = multiprocessing.get_all_start_methods()
    multiprocessing_enabled = True
    if "fork" in start_methods:
        multiprocessing.set_start_method("fork")
    else:
        multiprocessing_enabled = False
        logger.warning(
            "Fork multiprocessing is not supported, in-process dataloading will be used"
        )
    drop_last = run_opts.get("data_parallel_backend", False)
    for key in ["train", "valid", "test"]:
        opts = hparams[f"{key}_dataloader_opts"]
        if not multiprocessing_enabled:
            opts["num_workers"] = 0
        if opts["num_workers"] == 0 and "prefetch_factor" in opts:
            del opts["prefetch_factor"]
        opts["drop_last"] = drop_last


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



if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Check/adjust multiprocessing settings
    check_multiprocessing(hparams, run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from ljspeech_prepare import prepare_ljspeech
    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_ljspeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["save_folder"],
                "splits": hparams["splits"],
                "split_ratio": hparams["split_ratio"],
                "seed": hparams["seed"],
                "extract_features": ["audio_tokens"],
                "extract_features_opts": hparams["extract_features_opts"],
                "model_name": "toktts"
            },
        )

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Apply overfit test settings
    datasets = apply_overfit_test(hparams, datasets)

    # Trainer initialization
    tts_brain = TokTTSBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    tts_brain.sample_data = datasets["sample"]

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    tts_brain.fit(
        tts_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = tts_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # Save final checkpoint (fixed name)
    tts_brain.checkpointer.save_checkpoint(name="latest")
