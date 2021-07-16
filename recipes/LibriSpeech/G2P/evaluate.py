"""Recipe for evaluating a grapheme-to-phoneme system with librispeech lexicon.
The script may be use in isolation or in combination with Orion to fit
hyperparameters that do not require model retraining (e.g. Beam Search)
"""


from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import SaveableDataLoader
from train import dataio_prep
from hyperpyyaml import load_hyperpyyaml
from functools import partial
from types import SimpleNamespace
import itertools
import speechbrain as sb
import torch
import sys
import json
import logging

logger = logging.getLogger(__name__)


orion_is_available = False
try:
    import orion.client
    orion_is_available = True
except ImportError:
    logger.warn("Orion is not available")


class G2PEvaluator:
    """
    The G2P model evaluation wrapper

    Arguments
    ---------
    hparams: dict
        the dictionary from a parsed hyperparameter file
    device: str
        the device identifier
    model_state: dict
        a pre-loaded model state for a "warm start" if applicable
        - could be useful if hyperparameters have changed, but
        the same model can be reused from one run to the next
    """
    def __init__(self, hparams, device, model_state=None):
        self.hparams = SimpleNamespace(**hparams)
        self.overrides = overrides
        self.device = device
        self.modules = torch.nn.ModuleDict(self.hparams.modules).to(self.device)
        self.beam_searcher = self.hparams.beam_searcher.to(self.device)
        if model_state:
            self.hparams.model.load_state_dict(model_state)
        else:
            self.load()
        self.modules["model"].eval()

    def load(self):
        """
        Loads a model from a checkpoint
        """
        checkpointer = self.hparams.checkpointer
        checkpointer.recover_if_possible(
            device=torch.device(self.device)
        )

    def evaluate_batch(self, batch):
        """
        Evaluates the G2P model

        Arguments
        ---------
        batch: PaddedBatch
            A single batch of data, same as the kind of batch used
            for G2P training
        """
        batch = batch.to(self.device)
        if self.hparams.eval_mode == "sentence":
            hyps, scores = self._get_phonemes(
                batch.grapheme_encoded,
                batch.phoneme_encoded_bos
            )
        elif self.hparams.eval_mode == "word":
            hyps, scores = self._get_phonemes_wordwise(
                batch.grapheme_encoded,
                batch.phoneme_encoded_bos
            )

        ids = batch.id

        phns, phn_lens = batch.phn_encoded

        self.per_metrics.append(
            ids,
            hyps,
            phns,
            None,
            phn_lens,
            self.hparams.phoneme_encoder.decode_ndim,
        )

    def _get_phonemes(self, grapheme_encoded, phn_encoded_bos=None):
        if not phn_encoded_bos:
            phn_encoded_bos = (
                torch.zeros(
                    len(grapheme_encoded), 1
                ).to(self.device),
                torch.ones(len(grapheme_encoded))
            )
        p_seq, char_lens, encoder_out = self.hparams.model(
            grapheme_encoded=grapheme_encoded,
            phn_encoded=phn_encoded_bos,
        )
        return self.beam_searcher(encoder_out, char_lens)

    def _get_phonemes_wordwise(self, grapheme_encoded, phn_encoded_bos):
        for grapheme_item, phn_item, grapheme_len, phoneme_len in zip(
            grapheme_encoded, phn_encoded_bos
        ):
            words_batch = self._split_words_batch(grapheme_item, grapheme_len)
            hyps, scores = self._get_phonemes(words_batch)

    def _split_words_batch(self, graphemes, length):
        return PaddedBatch(
            {"grapheme_encoded": word}
            for word in self._split_words_seq(graphemes, length))

    def _split_words_seq(self, graphemes, length):
        space_index = self.hparams.graphemes.index(" ")
        word_boundaries = torch.where(graphemes == space_index)
        last_word_boundary = 0
        for word_boundary in word_boundaries:
            yield graphemes[last_word_boundary:word_boundary]
            last_word_boundary = word_boundary

        if last_word_boundary < length:
            yield graphemes[last_word_boundary:word_boundary]


    def evaluate_epoch(self, dataset, dataloader_opts=None):
        """
        Evaluates a single epoch

        Arguments
        ---------
        dataset: DynamicItemDataset
            a G2P dataset (same as the ones used for training)

        Returns
        -------
        metrics: dict
            Raw PER metrics
        """
        logger.info("Beginning evaluation")
        self.per_metrics = self.hparams.per_stats()
        dataloader = sb.dataio.dataloader.make_dataloader(
            dataset,
            **dict(dataloader_opts or {}, shuffle=True,
                   batch_size=self.hparams.eval_batch_size)
        )
        dataloader_it = iter(dataloader)
        if self.hparams.eval_batch_count is not None:
            dataloader_it = itertools.islice(dataloader_it, 0, self.hparams.eval_batch_count)
        for batch in dataloader_it:
            self.evaluate_batch(batch)
        return self.per_metrics.summarize()


if __name__ == "__main__":
    # CLI:

    # Parse the hyperparameter file
    search_hparam_file = sys.argv[0]
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    device = run_opts.get('device', 'cpu')
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Run the evaluation
    evaluator = G2PEvaluator(hparams, device)

    #Load the pretrained language model, if available:
    # We download and pretrain the tokenizer
    deps_pretrainer = hparams.get("deps_pretrainer")
    if deps_pretrainer:
        run_on_main(deps_pretrainer.collect_files)
        deps_pretrainer.load_collected(device=run_opts["device"])


    train_step = next(
        train_step for train_step in hparams['train_steps']
        if train_step['name'] == hparams["eval_train_step"])
    train, valid, test, _ = dataio_prep(hparams, train_step)
    datasets = {"train": train, "valid": valid, "test": test}
    dataset = datasets[hparams["eval_dataset"]]
    dataloader_opts = train_step.get(
            "dataloader_opts",
            hparams.get("dataloader_opts", {}))
    result = evaluator.evaluate_epoch(dataset, dataloader_opts)

    # Report the results
    if orion_is_available and hparams["eval_reporting"] == "orion":
        orion.client.report_objective(result["error_rate"])
    else:
        print(json.dumps(result))