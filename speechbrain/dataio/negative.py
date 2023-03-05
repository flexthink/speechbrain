"""Utilities for datasets with negative examples - useful for computing
triplet losses

Authors
  * Artem Ploujnkov 2023
"""
import speechbrain as sb
import torch
import copy
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from functools import partial
from speechbrain.utils import checkpoints


@checkpoints.register_checkpoint_hooks
class ReproducibleNegativeSampler:
    """A helper class to sample negative examples in a 
    reproducible manner
    
    Arguments
    ---------
    seed: int
        the random seed
    """
    def __init__(self, seed=42):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __call__(self, data_ids):
        """Obtains a new random sample
        
        Arguments
        ---------
        data_ids: list
            a list of data IDs
            
        Returns
        -------
        negative_data_ids: list
            the IDs of negative examples
        """
        negative_idx = torch.randperm(len(data_ids), generator=self.generator)
        return [
            data_ids[idx] for idx in negative_idx]
    
    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = self.generator.get_state()
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device  # Unused in here
        state = torch.load(path)
        self.generator.set_state(state)    


class NegativeEnhancedDataSet(FilteredSortedDynamicItemDataset):
    """A dataset enhanced with negative examples
    
    Arguments
    ---------
    from_dataset: speechbrain.dataio.dataset.DynamicItemDataset
        the source dataset

    sampler: ReproducibleNegativeSampler
        a sampler
    
    """
    def __init__(self, from_dataset, sampler=None):
        super().__init__(from_dataset, from_dataset.data_ids)
        self.pipeline_src = copy.deepcopy(from_dataset.pipeline)
        if not sampler:
            sampler = ReproducibleNegativeSampler()
        self.sampler = sampler
        self.negative_map = None
        self.data_ids_negative = None

    def sample_negative(self):
        """Resamples negative examples"""
        self.data_ids_negative = self.sampler(self.data_ids)
        self.negative_map = {
            data_id: negative_data_id
            for data_id, negative_data_id in zip(
                self.data_ids, self.data_ids_negative
            )
        }

    def ensure_negative_sample(self):
        """Samples negative examples if not already done"""
        if self.data_ids_negative is None:
            self.sample_negative()

def negative_pipeline(data_id, dataset, keys):
    dataset.ensure_negative_sample()
    negative_data_id = dataset.negative_map[data_id]
    data_point = dataset.data[negative_data_id]
    data = dataset.pipeline_src.compute_outputs({"id": negative_data_id, **data_point})
    for key in keys:
        yield data[key]


def add_negative(dataset, keys, sampler=None):
    """Enhances a dataset with negative keys
    
    Arguments
    ---------
    dataset: speechbrain.dataio.dataset.DynamicItemDataset
        a dataset
    keys: list
        a list of keys for which negative examples will be included
    sampler: callable
        the sampler to use

    Returns
    -------
    dataset_negative
        a dataset enhanced with negative examples
    """
    dataset_negative = NegativeEnhancedDataSet(dataset, sampler=sampler)
    negative_keys = [f"{key}_neg" for key in keys]
    dataset_negative.pipeline_src.set_output_keys(keys)
    dataset_negative_pipeline = partial(negative_pipeline, dataset=dataset_negative, keys=keys)
    dataset_negative_pipeline = sb.utils.data_pipeline.takes("id")(dataset_negative_pipeline)
    dataset_negative_pipeline = sb.utils.data_pipeline.provides(*negative_keys)(dataset_negative_pipeline)
    dataset_negative.add_dynamic_item(dataset_negative_pipeline)
    return dataset_negative

    