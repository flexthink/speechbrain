#!/usr/bin/env/python3
"""Recipe for training the MadMixture model, a generic multimodal \
fusion model for speech and text

Authors
 * Artem Ploujnikov 2022
"""

import sys
import torch
import logging
import speechbrain as sb
import os
from hyperpyyaml import load_hyperpyyaml
from librispeech_prepare import prepare_librispeech, LibriSpeechMode
from speechbrain.utils.distributed import run_on_main
from collections import namedtuple

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class MadMixtureBrain(sb.Brain):
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
            At training time it returns predicted seq2seq log probabilities.
            If needed it also returns the ctc output log probabilities.
            At validation/test time, it returns the predicted tokens as well.
        """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        feats, lengths, context = self.prepare_features(stage, batch)
        latents, alignments, enc_out, rec = self.modules.model.train_step(feats, lengths, context)
        return MadMixturePredictions(latents,alignments, enc_out, rec, feats)

    def prepare_features(self, stage, batch):
        """Prepare features for computation on-the-fly

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        wavs : tuple
            The input signals (tensor) and their lengths (tensor).
        """
        wavs, wav_lens = batch.sig

        #TODO: This can be made more modular

        # Feature computation and normalization
        feats_audio = self.hparams.compute_features(wavs)
        feats_audio = self.modules.normalize(feats_audio, wav_lens)
        
        feats_char_emb = self.hparams.char_emb(batch.char_encoded.data)
        feats_phn_emb = self.hparams.phn_emb(batch.phn_encoded.data)

        feats = {
            "audio": feats_audio,
            "char": feats_char_emb,
            "phn": feats_phn_emb,
        }

        lengths = {
            "audio": wav_lens,
            "char": batch.char_encoded.lengths,
            "phn": batch.phn_encoded.lengths,
        }

        context = {
            "audio": feats_audio,
            "char_emb": feats_char_emb,
            "phn_emb": feats_phn_emb,
        }

        return feats, lengths, context

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : MadMixturePredictions
            The output from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        latents, alignments, enc_out, rec, feats = predictions
        rec_loss = {
            key: spec["rec_loss"](rec[key], feats[key])
            for key, spec in self.hparams.modalities.items()
        }
        rec_loss_weighted = {
            key: self.hparams.modalities[key]["rec_loss_weight"] * mod_rec_loss
            for key, mod_rec_loss in rec_loss.items()
        }

        rec_loss_total = sum(rec_loss_weighted.values())

        loss = rec_loss_total
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

        # Summarize the statistics from the stage for record-keeping.
        else:
            pass

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

MadMixturePredictions = namedtuple(
    "MadMixturePredictions",
    [
        "latents",
        "alignments",
        "enc_out",
        "rec",
        "feats"
    ]
)

LIBRISPEECH_OUTPUT_KEYS = [
    "wrd_count",
    "sig",
    "wrd_start",
    "wrd_end",
    "phn_start",
    "phn_end",
    "wrd",
    "char",
    "phn",
]

LIBRISPEECH_OUTPUT_KEYS_DYNAMIC = LIBRISPEECH_OUTPUT_KEYS + [
    "phn_encoded",
    "phn_encoded_bos",
    "phn_encoded_eos",
    "char_encoded",
    "char_encoded_bos",
    "char_encoded_eos"
]

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
    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    def sequence_pipeline(prefix, label_encoder):
        @sb.utils.data_pipeline.takes(f"{prefix}")
        @sb.utils.data_pipeline.provides(
            f"{prefix}",
            f"{prefix}_list",
            f"{prefix}_encoded_bos",
            f"{prefix}_encoded_eos",
            f"{prefix}_encoded"
        )
        def pipeline_fn(seq):
            yield seq
            seq_list = list(seq)
            yield seq_list
            tokens_list = label_encoder.encode_sequence_torch(seq_list)
            yield tokens_list
            tokens_bos = label_encoder.prepend_bos_index(tokens_list)
            yield tokens_bos
            tokens_eos = label_encoder.append_eos_index(tokens_list)
            yield tokens_eos
            tokens = torch.LongTensor(tokens_list)
            yield tokens
        return pipeline_fn

        

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
  
    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    char_encoder = init_sequence_encoder(hparams, "char")
    char_pipeline = sequence_pipeline("char", char_encoder)
    phn_encoder = init_sequence_encoder(hparams, "phn")
    phn_pipeline = sequence_pipeline("phn", phn_encoder)
    dynamic_items = [audio_pipeline, char_pipeline, phn_pipeline]

    for dataset in data_info:
        # Load the full LibriSpeech dataset
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=dynamic_items,
            output_keys=LIBRISPEECH_OUTPUT_KEYS_DYNAMIC
        )

        # Use the curriculum sampler to reduce the dataset's complexity
        if hparams["curriculum_enabled"]:
            curriculum = hparams["curriculum"]
            logger.info(
                "Using curriculum sampling with %d-%d words, %d samples",
                curriculum["min_words"],
                curriculum["max_words"],
                curriculum["num_samples"]
            )
            curriculum_generator = torch.Generator()
            curriculum_generator.manual_seed(hparams["seed"])
            dynamic_dataset = sb.dataio.curriculum.CurriculumSpeechDataset(
                from_dataset=dynamic_dataset,
                min_words=curriculum["min_words"],
                max_words=curriculum["max_words"],
                num_samples=curriculum["num_samples"],
                sample_rate=hparams["sample_rate"],
                generator=curriculum_generator,
            )
        else:
            logger.info("Curriculum sampling is disabled, using the complete dataset")
        dynamic_dataset.set_output_keys(LIBRISPEECH_OUTPUT_KEYS_DYNAMIC)

        for dynamic_item in dynamic_items:
            dynamic_dataset.add_dynamic_item(dynamic_item)
        datasets[dataset] = dynamic_dataset
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="duration")
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
    return datasets


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
    if not os.path.isfile(file_name):
        raise ValueError(f"Token file {file_name} not found")
    with open(file_name) as token_file:
        return [line.strip() for line in token_file if line]
    

def init_sequence_encoder(hparams, prefix):
    encoder = hparams[f"{prefix}_label_encoder"]
    token_list_file_name = hparams[f"{prefix}_list_file"]
    tokens = read_token_list(token_list_file_name)
    encoder.add_bos_eos()
    encoder.add_unk()
    encoder.update_from_iterable(tokens)
    return encoder



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

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "alignments_folder": hparams["data_folder_alignments"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.json",
            "skip_prep": hparams["skip_prep"],
            "mode": LibriSpeechMode.ALIGNMENT
        },
    )

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Trainer initialization
    madmixture_brain = MadMixtureBrain(
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
    madmixture_brain.fit(
        madmixture_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = madmixture_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
