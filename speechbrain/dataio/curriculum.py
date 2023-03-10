"""
Utilities for curriculum learning

Authors
* Artem Ploujnikov 2022
"""
import os
import torch
import speechbrain as sb
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)

from speechbrain.dataio.dataset import (
    DynamicItemDataset,
    FilteredSortedDynamicItemDataset,
)


SAMPLE_OUTPUTS = [
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


class CurriculumSpeechDataset(DynamicItemDataset):
    """A derivative dynamic dataset that allows to perform
    curriculum learning over a speech dataset with phoneme
    alignments similar to LibriSpeech-Alignments. The dataset
    selects sub-samples within the specified length range in words

    Arguments
    ---------
    from_dataset: DynamicItemDataset
        a base dataset compatible with alignments-enhanced LibriSpeech
    min_words: int
        the minimum number of words to sample from each dataset item
    max_words: int
        the maximum number of words to sample from each dataset item
    num_samples: int
        the number of samples per epoch
    sample_rate: int
        the audio sampling rate, in Hertz
    generator: torch.Generator
        a random number generator (optional). A custom generator may
        be supplied for reproducibility or fofr unit tests.
    """

    def __init__(
        self,
        from_dataset,
        min_words=None,
        max_words=None,
        num_samples=None,
        sample_rate=16000,
        generator=None,
    ):
        super().__init__(
            data=from_dataset.data, use_existing_id=True)
        self.generator = generator
        self.base_dataset = sample(from_dataset, num_samples, generator)
        self.data_ids = self.base_dataset.data_ids
        self.min_words = min_words
        self.max_words = max_words
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.data_id_indices = {
            data_id: idx for idx, data_id in enumerate(self.data_ids)
        }
        self._pipeline_is_setup = False
        if (
            min_words is not None
            or max_words is not None
            or num_samples is not None
        ):
            self.sample_segments()
        self.setup_pipeline()
        self.pipeline = PipelineWrapper(self.pipeline, SAMPLE_OUTPUTS)
            
    def sample_segments(self, dataset=None):
        """Samples parts of the audio file at specific word boundaries

        Arguments
        ---------
        datset: DynamicItemDataset
            the dataset from which to sample
        """
        # Exclude samples than have fewer
        # words than the minimum
        if dataset is None:
            dataset = self.base_dataset
        dataset = dataset.filtered_sorted(
            key_min_value={"wrd_count": self.min_words}
        )
        keys = ["wrd_count", "wrd_start", "wrd_end"]
        with dataset.output_keys_as(keys):
            wrd_count = torch.tensor(self._pluck("wrd_count"))
            wrd_start = self._pluck("wrd_start")
            wrd_end = self._pluck("wrd_end")

        # Randomly sample word counts in the
        # range form num_words to last_words
        self.sample_word_counts = torch.randint(
            low=self.min_words,
            high=self.max_words + 1,
            size=(len(dataset),),
            generator=self.generator,
        )

        # Sample relative offsets, from 0.0 to 1.0.
        # 0.0 corresponds to the beginning of the
        # utterance, where as 1.0 represents wrd_count - n
        # where n is the sampled word count
        sample_offsets_rel = torch.rand(len(dataset), generator=self.generator)

        # Determine the maximum possible offsets
        max_offset = wrd_count - self.sample_word_counts
        self.wrd_offset_start = (sample_offsets_rel * max_offset).floor().int().clamp(0)
        self.wrd_offset_end = (self.wrd_offset_start + self.sample_word_counts)
        self.wrd_offset_end = torch.maximum(self.wrd_offset_end, self.wrd_offset_start + 1)
        self.wrd_offset_end = torch.minimum(self.wrd_offset_end, wrd_count - 1)
        sample_start = torch.tensor(
            [item[idx] for item, idx in zip(wrd_start, self.wrd_offset_start)]
        )
        sample_end = torch.tensor(
            [item[idx - 1] for item, idx in zip(wrd_end, self.wrd_offset_end)]
        )
        sample_start_idx = time_to_index(sample_start, self.sample_rate)
        sample_end_idx = time_to_index(sample_end, self.sample_rate)
        self.sample_start_idx = sample_start_idx
        self.sample_end_idx = sample_end_idx

    def _pluck(self, key):
        """Retrieves a list of values of the specified key from
        all data items in the dataset

        Arguments
        ---------
        key: str
            the key

        Returns
        -------
        result: list
            the resulting list"""
        return [self.data[data_id][key] for data_id in self.data_ids]

    def setup_pipeline(self):
        """Sets up the dynamic pipeline to sample from the dataset
        using the previously generated samples"""

        @sb.utils.data_pipeline.takes(
            "id",
            "wav",
            "phn_start",
            "phn_end",
            "wrd_start",
            "wrd_end",
            "wrd",
            "phn",
        )
        @sb.utils.data_pipeline.provides(
            "_wrd_count",
            "_sig",
            "_wrd_start",
            "_wrd_end",
            "_phn_start",
            "_phn_end",
            "_wrd",
            "_char",
            "_phn",
        )
        def cut_sample(
            data_id, wav, wrd_start, wrd_end, phn_start, phn_end, wrd, phn
        ):
            idx = self.data_id_indices[data_id]
            # wrd_count
            yield self.sample_word_counts[idx].item()
            sample_start_idx = self.sample_start_idx[idx]
            sample_end_idx = self.sample_end_idx[idx]
            sig = sb.dataio.dataio.read_audio(wav)
            sig = sig[sample_start_idx:sample_end_idx]
            # sig
            yield sig
            wrd_offset_start = self.wrd_offset_start[idx]
            wrd_offset_end = self.wrd_offset_end[idx]
            # wrd_start
            yield cut_offsets(wrd_start, wrd_offset_start, wrd_offset_end)
            # wrd_end
            yield cut_offsets(wrd_end, wrd_offset_start, wrd_offset_end)
            # phn_start
            phn_start, phn_from, phn_to = cut_offsets_rel(
                wrd_start, phn_start, wrd_offset_start, wrd_offset_end
            )
            yield phn_start
            # phn_end
            phn_end, _, _ = cut_offsets_rel(
                wrd_end, phn_end, wrd_offset_start, wrd_offset_end
            )
            yield phn_end
            # wrd
            wrd_sample = wrd[wrd_offset_start:wrd_offset_end]
            yield wrd_sample
            yield " ".join(wrd_sample).upper()
            phn = phn[phn_from:phn_to]
            yield phn

        self.pipeline.add_dynamic_item(cut_sample)


def sample(base_dataset, num_samples, generator=None):
    """Retrieves a sample of the base dataset
    
    Arguments
    ---------
    base_dataset: DynamicItemDataset
        a base dataset
    num_samples: int
        the number of samples to include
    generator: torch.Generator
        a random number generator (optional)
        
    Returns
    -------
    dataset: FilteredSortedDynamicItemDataset
        a random sample of the dataset    
    """
    dataset = base_dataset
    if num_samples is not None and num_samples != len(base_dataset):
        sample_indexes = torch.multinomial(
            torch.ones(len(dataset)) / len(dataset),
            num_samples=num_samples,
            replacement=num_samples > len(base_dataset),
            generator=generator
        )
        sample_data_ids = [
            dataset.data_ids[idx]
            for idx in sample_indexes
        ]
    
        dataset = FilteredSortedDynamicItemDataset(
            from_dataset=dataset,
            data_ids=sample_data_ids
        )

    return dataset

PIPELINE_WRAPPER_ATTRS = {"pipeline", "key_map"}

class PipelineWrapper:
    """A pipeline wrapper that makes it possible to replace
    static outputs with dynamic ones. The trick is to output an
    item with the desired key prefixed with a '_'. The '_' will
    br removed in the output


    Arguments
    ---------
    pipeline: torch.tensor
        the original pipeline
    replace_keys: enumerable
        the list of keys that will be replaced

    """

    def __init__(self, pipeline, replace_keys):
        self.pipeline = pipeline
        self.key_map = {key: f"_{key}" for key in replace_keys}

    def compute_outputs(self, data):
        """Computes the output

        Arguments
        ---------
        data: dict
            the static data

        Returns
        -------
        result: dict
            the pipeline output
        """
        result = self.pipeline.compute_outputs(data)
        for key, key_r in self.key_map.items():
            if key_r in result:
                result[key] = result[key_r]
                del result[key_r]
        return result

    def set_output_keys(self, keys):
        """Sets the keys to be output by the pipeline

        Arguments
        ---------
        keys: enumerable
            a list of keys
        """
        keys_r = {self.key_map.get(key, key) for key in keys}
        self.pipeline.set_output_keys(keys_r)

    def __getattr__(self, name):
        """Delegates attribute calls to the underlying pipeline
        
        Arguments
        ---------
        name: str
            the attribute name

        Returns
        -------
        value: object
            the attribute value
        """
        if name in PIPELINE_WRAPPER_ATTRS:
            if name not in self.__dict__:
                raise AttributeError()
            return self.__dict__[name]

        return getattr(self.pipeline, name)


def time_to_index(times, sample_rate):
    """Converts a collection of time values to a list of
    wave array indexes at the specified sample rate

    Arguments
    ---------
    times: enumerable
        a list of time values
    sample_rate: int
        the sample rate (in hertz)

    Returns
    -------
    result: list
        a collection of indexes
    """

    if not torch.is_tensor(times):
        times = torch.tensor(times)

    return (times * sample_rate).floor().int().tolist()


def cut_offsets(offsets, start, end):
    """Given an array of offsets (e.g. word start times),
    returns a segment of it from <start> to <end> re-computed
    to begin at 0

    Arguments
    ---------
    offsets: list|torch.tensor
        a list or tensor of offsets

    start: int
        the starting index

    end: int
        the final index

    Returns
    -------
    result: list
        the re-calculated offset list
    """
    segment = offsets[start:end]
    if not torch.is_tensor(segment):
        segment = torch.tensor(segment)
    return (segment - segment[0]).tolist()


def cut_offsets_rel(offsets, ref_offsets, start, end):
    """Given a sequence of offsets (e.g. phoneme offsets)
    and a reference sequence (e.g. sequence of words), finds
    the range in <offsets> corresponding to the specified range
    in <ref_offsets>

    Arguments
    ---------
    offsets: list|torch.Tensor
        a collection of offsets

    ref_offsets: list|torch.Tensor
        reference offsets

    Returns
    -------
    result: list
        the corresponding values in offsets
    start: int
        the start index
    end: int
        the end index
    """
    if not torch.is_tensor(offsets):
        offsets = torch.tensor(offsets)
    if not torch.is_tensor(ref_offsets):
        ref_offsets = torch.tensor(ref_offsets)
    start_value = ref_offsets[start].item()
    end_value = ref_offsets[end].item()
    condition = (offsets >= start_value) & (offsets < end_value)
    result = offsets[condition]
    result -= result[0].item()
    idx = condition.nonzero()
    return result.tolist(), idx.min().item(), idx.max().item() + 1

@checkpoints.register_checkpoint_hooks
class CurriculumController:
    """Provides control for the curriculum dataset from the training
    process"""

    def __init__(self):
        self.dataset = None
        self.generator = None

    def bind(self, dataset):
        """Binds this controller to a dataset
        
        Arguments
        ---------
        dataset: CurriculumSpeechDataset
            a curriculum dataset
        """
        self.dataset = dataset
        self.generator = dataset.generator

    def resample(self, min_words=None, max_words=None, num_samples=None):
        """Resamples the dataset
        
        Arguments
        ---------
        min_words: int
            the minimum number of words. If omitted, the value is not changed
        
        max_words: int
            the maximum number of words. If omitted, the value is not changed
        
        num_samples: int
            the number of samples. If omitted, the value is not changed
        """
        if self.dataset is None:
            raise ValueError("The curriculum controller is unbound")
        if min_words is None:
            min_words = self.dataset.min_words
        if max_words is None:
            max_words = self.dataset.max_words
        if num_samples is None:
            num_samples = self.dataset.num_samples
        
        self.dataset.min_words = min_words
        self.dataset.max_words = max_words
        self.dataset.num_samples = num_samples
        self.dataset.sample_segments()

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {"generator": self.generator.get_state()}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device  # Unused in here
        state = torch.load(path)
        self.generator.set_state(state["generator"])

@checkpoints.register_checkpoint_hooks
class Curriculum:
    """A helper class to define a curriculum
    
    Arguments
    ---------
    steps: list
        a list of dicts similar to the following
        [
            {"epoch": 1, "max_words": 3},
            {"epoch": 5, "max_words": 5},
            {"epoch": 10, "max_words": 5},
        ]
    controller: CurriculumController
        a curriculum controller
    """

    def __init__(self, steps, controller=None):
        self.steps = sorted(
            [
                {"epoch": 0, **step}
                for step in steps
            ],
            key=lambda step: step["epoch"]
        )
        if controller is None:
            controller = CurriculumController()
        self.controller = controller
        self.step_id = None

    def apply(self, epoch):
        """Finds the step corresponding to the specified
        epoch
        
        Arguments
        ---------
        epoch: int
            the epoch number
            
        Returns
        -------
        step_id: int
            the step ID / number (starting at 1)
        step: dict
            the step configuration
        """
        step_id, step = self.find_step(epoch)
        if step_id is None:
            logging.warn(
                "Unable to find a curriculum step epoch %d",
                epoch
            )
            return None, None
        kwargs = {**step}
        del kwargs["epoch"]
        self.controller.resample(**kwargs)
        self.step_id = step_id
        return step_id, step
    
    def find_step(self, epoch):
        """Finds the step corresponding to the specified
        epoch
        
        Arguments
        ---------
        epoch: int
            the epoch number
            
        Returns
        -------
        step_id: int
            the step ID / number (starting at 1)
        step: dict
            the step configuration
        """
        return next(
            ((step_id, step) 
             for step_id, step in reversed(
                list(enumerate(self.steps, start=1))
             )
             if epoch >= step["epoch"]),
            (None, None)
        )
    
    def bind(self, dataset):
        """Binds the underlying controller to a dataset
        
        Arguments
        ---------
        dataset: CurriculumSpeechDataset
            a curriculum dataset
        """
        self.controller.bind(dataset)

    def save_dataset(
            self,
            path=None,
            keys=None
        ):
        """Saves the dataset contents for future reference, analysis
        and debugging
        
        Arguments
        ---------
        dataset_key: str
            a string key to identify the dataset
        path: str
            the filesystem path
        keys: list
            the data keys to output
        """
        dataset = self.controller.dataset
        if path is None:
            path = "."
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, f"data-{self.step_id}.json")
        if not os.path.exists(file_name):
            if keys is not None:
                with dataset.output_keys_as(keys):
                    dataset.to_json(file_name)
            else:
                dataset.to_json(file_name)

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""        
        self.controller.save(path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        self.controller.load(path, end_of_epoch, device)
