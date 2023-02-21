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
from collections import namedtuple
from speechbrain.dataio.dataset import apply_overfit_test

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class MadMixtureBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs the computation of the MadMixture model latents and
        reconstructions for each modality

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : MadMixturePredictions
            A namedtuple containing the following
                latents: dict
                    latent representations for each modality
                alignments: dict
                    alignment matrices for all aligned modalities
                enc_out: dict
                    raw encoder outputs (before alignment)
                rec: dict
                    modality reconstructions


        """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        feats, lengths, context = self.prepare_features(stage, batch)
        latents, alignments, enc_out, rec, transfer_rec, out_context, length_preds, lengths_latent, lengths_input = self.modules.model.train_step(
            feats, lengths, context, transfer=self.hparams.transfer_loss_enabled)
        return MadMixturePredictions(
            latents, alignments, enc_out, rec, transfer_rec, out_context, length_preds, feats, lengths, lengths_latent, lengths_input)
    
    def fit_batch(self, batch):
        result = super().fit_batch(batch)
        if (
            self.hparams.enable_train_metrics
            and self.hparams.use_tensorboard
            and (
                self.step == 1
                or self.step % self.hparams.train_log_interval == 0
            )
        ):
            self.log_batch()
        return result

    
    
    def log_batch(self):
        stats = {
            "lr": self.optimizer.param_groups[0]["lr"],
            **self.loss_metric.summarize(field="average")
        }
        self.hparams.tensorboard_train_logger.log_stats(
            stats_meta={"step": self.step}, train_stats=stats
        )


    
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
        feats = {}
        lengths = {}
        feats_audio = None
        if self.hparams.audio_enabled:
            feats_audio = self.hparams.compute_features(wavs)
            if self.hparams.compute_features_transpose:
                feats_audio = feats_audio.transpose(-1, -2)
            for norm in self.modules.normalize:
                feats_audio = norm(feats_audio, wav_lens)        
            feats["audio"] = feats_audio
            lengths["audio"] = wav_lens


        # TODO: Embeddings are computed twice, avoid this
        feats_char_emb = None
        feats_phn_emb = None
        if self.hparams.char_enabled:
            feats["char"] = batch.char_encoded_eos.data
            lengths["char"] = batch.char_encoded_eos.lengths
            feats_char_emb = self.hparams.char_emb(batch.char_encoded_bos.data)
        if self.hparams.phn_enabled:
            feats["phn"] = batch.phn_encoded_eos.data
            lengths["phn"] = batch.phn_encoded_eos.lengths
            feats_phn_emb = self.hparams.phn_emb(batch.phn_encoded_bos.data)

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
        loss = self.hparams.compute_cost(
            inputs=predictions.feats,
            latents=predictions.latents,
            alignments=predictions.alignments,
            rec=predictions.rec,
            length=predictions.lengths,
            transfer_rec=predictions.transfer_rec,
            out_context=predictions.out_context,
            length_preds=predictions.length_preds,
            length_latent=predictions.lengths_latent,
            length_input=predictions.lengths_input,
        )

        if self.hparams.enable_train_metrics and self.step % self.hparams.train_metrics_interval == 0:
            self.loss_metric.append(
                batch.snt_id,
                inputs=predictions.feats,
                latents=predictions.latents,
                alignments=predictions.alignments,
                rec=predictions.rec,
                length=predictions.lengths,
                transfer_rec=predictions.transfer_rec,
                out_context=predictions.out_context,
                length_latent=predictions.lengths_latent,
                length_input=predictions.lengths_input,
                reduction="batch"
            )
        return loss
    
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, performing the various tasks configured in
        hparams (e.g. same-modality reconstruction, cross-modality evaluation,
        etc)

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """

        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        self.evaluator.append(
            ids=batch.snt_id,
            inputs=out.feats,
            latents=out.latents,
            alignments=out.alignments,
            lengths=out.lengths,
            targets=out.feats,
            out_context=out.out_context
        )        
        return loss.detach().cpu()
    
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
        self.loss_metric = sb.utils.metric_stats.MultiMetricStats(
            metric=self.hparams.compute_cost.details,
            batch_eval=True
        )
        self.stage_vis_sample = (
            self.vis_sample[stage]
            if hasattr(self, "vis_sample")
            else None
        )
        self.evaluator = self.hparams.evaluator()
        self.evaluator.use_vis_sample(self.stage_vis_sample)

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
            # NOTE: Checkpoints can be skipped during debugging to avoid undue
            # stress on the SSD
            if not self.hparams.skip_checkpoint:
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
                )

            # Write evaluation reports for the epoch
            self.evaluation_report()

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def evaluation_report(self):
        """Outputs the evaluation report"""
        report_path = os.path.join(
            self.hparams.reports_folder,
            str(self.hparams.epoch_counter.current)
        )
        self.evaluator.report(report_path)

    def use_vis_sample(self, data_ids):
        """
        Arguments
        ---------
        data_ids: dict
            a key -> enumerable dictionary
        """
        self.vis_sample = {
            sb.Stage[key.upper()]: stage_data_ids
            for key, stage_data_ids in data_ids.items()
        }

MadMixturePredictions = namedtuple(
    "MadMixturePredictions",
    [
        "latents",
        "alignments",
        "enc_out",
        "rec",
        "transfer_rec",
        "out_context",
        "length_preds",
        "feats",
        "lengths",
        "lengths_latent",
        "lengths_input",
    ]
)

LIBRISPEECH_OUTPUT_KEYS = [
    "snt_id",
    "wrd_count",
    "sig",
    "wrd_start",
    "wrd_end",
    "phn_start",
    "phn_end",
    "wrd",
    "char",
    "phn",
    "unk_count"
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
        sig = sb.dataio.dataio.read_audio(wav) * hparams["amp_multiplier"]
        return sig

    def sequence_pipeline(prefix, label_encoder):
        @sb.utils.data_pipeline.takes(f"_{prefix}")
        @sb.utils.data_pipeline.provides(
            f"{prefix}",
            f"{prefix}_encoded",
            f"{prefix}_encoded_bos",
            f"{prefix}_encoded_eos",
        )
        def pipeline_fn(seq):
            seq_list = list(seq)
            yield seq_list
            tokens_list = label_encoder.encode_sequence_torch(seq_list)
            tokens = torch.LongTensor(tokens_list)
            yield tokens
            tokens_bos = label_encoder.prepend_bos_index(tokens_list)
            yield tokens_bos
            tokens_eos = label_encoder.append_eos_index(tokens_list)
            yield tokens_eos
        return pipeline_fn

        

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
  
    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    # Define samples for visualizations
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_annotation"],
    }
    if not hparams["overfit_test"]:
        data_info.update({
            "valid": hparams["valid_annotation"],
            "test": hparams["test_annotation"],
        })

    char_encoder = init_sequence_encoder(hparams, "char")
    char_pipeline = sequence_pipeline("char", char_encoder)
    phn_encoder = init_sequence_encoder(hparams, "phn")
    phn_pipeline = sequence_pipeline("phn", phn_encoder)
    dynamic_items = [audio_pipeline, char_pipeline, phn_pipeline]
    curriculum = hparams.get("curriculum")

    for dataset in data_info:
        # Load the full LibriSpeech dataset
        key_max_value = {
            "unk_count": 0,
        }
        key_min_value=None
        if hparams["curriculum_enabled"]:            
            key_min_value = {
                "wrd_count": curriculum["max_words"]            
            }
        key_test = {}
        if hparams["filter_spk_id"]:
            key_test["spk_id"] = hparams["filter_spk_id"]

        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            data_info[dataset],
            replacements={"data_root": data_folder},
        ).filtered_sorted(
            key_min_value=key_min_value,
            key_max_value=key_max_value,
            key_test=key_test
        )
        dynamic_dataset.set_output_keys(LIBRISPEECH_OUTPUT_KEYS)

        # Use the curriculum sampler to reduce the dataset's complexity
        if hparams["curriculum_enabled"]:
            curriculum = hparams["curriculum"]
            logger.info(
                "Using curriculum sampling with %d-%d words, %d samples",
                curriculum["min_words"],
                curriculum["max_words"],
                curriculum["num_samples"][dataset]
            )
            curriculum_generator = torch.Generator()
            curriculum_generator.manual_seed(hparams["seed"])
            dynamic_dataset = sb.dataio.curriculum.CurriculumSpeechDataset(
                from_dataset=dynamic_dataset,
                min_words=curriculum["min_words"],
                max_words=curriculum["max_words"],
                num_samples=curriculum["num_samples"][dataset],
                sample_rate=hparams["sample_rate"],
                generator=curriculum_generator,
            )
        else:
            logger.info("Curriculum sampling is disabled, using the complete dataset")
        for dynamic_item in dynamic_items:
            dynamic_dataset.add_dynamic_item(dynamic_item)
        dynamic_dataset.set_output_keys(LIBRISPEECH_OUTPUT_KEYS_DYNAMIC)
        datasets[dataset] = dynamic_dataset
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False
    datasets = apply_overfit_test(
        hparams, datasets)

    vis_samples = {
        key: select_vis_sample(dynamic_dataset, hparams)
        for key, dynamic_dataset in datasets.items()    
    }
    

    # Apply the sort order. Sorting by duration can help reduce
    # zero-padding. Such sorting is not applicable for overfit
    # tests
    if hparams["overfit_test"]:
        logger.info(
            "Performing an overfit test with %d samples used, %d per epoch "
            "- sorting will be ignored",
            hparams["overfit_test_sample_count"],
            hparams["overfit_test_epoch_data_count"]
        )
    elif hparams["sorting"] == "ascending":
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
    return datasets, vis_samples


def select_vis_sample(dataset, hparams):
    """Selects a sample of the specified size
    
    Arguments
    ---------
    dataset: DynamicItemDataset
        the dataset from which to select the sample
    hparams: dict 
        hyperparameters
    """
    sample_size = hparams["eval_vis_sample_size"]
    generator = torch.Generator()
    generator.manual_seed(hparams["seed"])
    indexes = torch.randperm(len(dataset.data_ids), generator=generator)[:sample_size]
    return [dataset.data_ids[idx] for idx in indexes]


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
        return [line.strip("\r\n") for line in token_file if line]
    

def init_sequence_encoder(hparams, prefix):
    """Initialize a sequence encoder
    
    Arguments
    ---------
    hparams: dict
        parsed hyperparameters
    prefix: str
        the prefix to be prepended to hyperparameter keys, per the naming
        convention
        
        {prefix}_label_encoder: the hparams key for the label encoder
        {prefix}_list_file: 
    
    Returns
    -------
    encoder: speechbrain.dataio.encoder.TextEncoder
        an encoder instance"""
    encoder = hparams[f"{prefix}_label_encoder"]
    token_list_file_name = hparams[f"{prefix}_list_file"]
    tokens = read_token_list(token_list_file_name)
    encoder.add_bos_eos()
    encoder.add_unk()
    encoder.update_from_iterable(tokens, sequence_input=False)
    return encoder

def check_tensorboard(hparams):
    """Checks whether Tensorboard is enabled and initializes the logger if it is

    Arguments
    ---------
    hparams: dict
        the hyperparameter dictionary
    """
    if hparams["use_tensorboard"]:
        try:
            from speechbrain.utils.train_logger import TensorboardLogger

            hparams["tensorboard_train_logger"] = TensorboardLogger(
                hparams["tensorboard_logs"]
            )
        except ImportError:
            logger.warning(
                "Could not enable TensorBoard logging - TensorBoard is not available"
            )
            hparams["use_tensorboard"] = False


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Check whether Tensorboard is available and enabled
    check_tensorboard(hparams)

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
    datasets, vis_samples = dataio_prepare(hparams)

    # Trainer initialization
    madmixture_brain = MadMixtureBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    madmixture_brain.use_vis_sample(vis_samples)

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
    
    if not hparams["overfit_test"]:
        # Load best checkpoint for evaluation
        test_stats = madmixture_brain.evaluate(
            test_set=datasets["test"],
            min_key="loss",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
