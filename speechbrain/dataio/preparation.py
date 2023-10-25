"""This file contains utilities for preprocessing of features, particularly
using neural models

Authors
 * Artem Ploujnikov 2023
"""
import torch
import numpy as np
import math
import speechbrain as sb
from pathlib import Path
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.batch import undo_batch
from speechbrain.utils.data_pipeline import DataPipeline
from tqdm.auto import tqdm


class FeatureExtractor:
    """A utility class for pipeline-based feature extraction

    Arguments
    ---------
    save_path: str|path-like
        the path where the preprocessed features will be saved

    id_key: str
        the key within the batch that will be used as an identifier

    save_format: str|callable
        the format in which prepared features will be saved

    device: str|torch.Device
        the device on which operations will be run

    dataloader_opts: dict
        parameters to be passed to the data loader (batch size, etc)
    
    dynamic_items : list
        Configuration for the dynamic items produced when fetching an example.
        List of DynamicItems or dicts with the format::
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides

    """
    def __init__(
            self,
            save_path,
            src_keys,
            id_key="id",
            save_format="npy",
            device="cpu",
            dataloader_opts=None,
            dynamic_items=None,
            description=None
    ):
        if not dataloader_opts:
            dataloader_opts = {}
        self.id_key = id_key
        self.save_path = save_path
        self.src_keys = src_keys
        self.id_key = id_key
        self.dataloader_opts = dataloader_opts
        if callable(save_format):
            self.save_fn = save_format
        elif save_format in SAVE_FORMATS:
            self.save_fn = SAVE_FORMATS[save_format]
        else:
            raise ValueError(f"Unsupported save_format: {save_format}")
        self.device = device
        self.pipeline = DataPipeline(
            static_data_keys=src_keys,
            dynamic_items=dynamic_items or []
        )
        self.description = description

    def extract(self, dataset):
        """Runs the preprocessing operation

        Arguments
        ---------
        dataset: dict|speechbrain.dataio.dataset.DynamicItemDataset
            the dataset to be saved
        """
        if isinstance(dataset, dict):
            dataset = DynamicItemDataset(dataset)
        dataset.set_output_keys(self.src_keys + [self.id_key])
       
        dataloader = make_dataloader(dataset, **self.dataloader_opts)
        batch_size = self.dataloader_opts.get("batch_size", 1)
        batch_count = int(math.ceil(len(dataset) / batch_size))
        for batch in tqdm(dataloader, total=batch_count, desc=self.description):
            batch = batch.to(self.device)
            self.process_batch(batch)

    def process_batch(self, batch):
        """Processes a batch of data
        
        Arguments
        ---------
        batch: speechbrain.dataio.batch.PaddedBatch
            a batch
        """
        batch_dict = batch.as_dict()
        ids = batch_dict[self.id_key]
        features = self.pipeline.compute_outputs(batch_dict)

        for item_id, item_features in zip(ids, undo_batch(features)):
            self.save_fn(item_id, item_features, save_path=self.save_path)

    def add_dynamic_item(self, func, takes=None, provides=None):
        """Adds a dynamic item to be output

        Two calling conventions. For DynamicItem objects, just use:
        add_dynamic_item(dynamic_item).
        But otherwise, should use:
        add_dynamic_item(func, takes, provides).

        See `speechbrain.utils.data_pipeline`.

        Arguments
        ---------
        func : callable, DynamicItem
            If a DynamicItem is given, adds that directly. Otherwise a
            DynamicItem is created, and this specifies the callable to use. If
            a generator function is given, then create a GeneratorDynamicItem.
            Otherwise creates a normal DynamicItem.
        takes : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single arg can be given directly.
        provides : str
            Unique key or keys that this provides.
        """
        self.pipeline.add_dynamic_item(func, takes, provides)

    def set_output_features(self, keys):
        """Sets the features to be output"""
        self.pipeline.set_output_keys(keys)


def save_pt(item_id, data, save_path):
    """Saves the data in the PyTorch format (one file per sample)

    Arguments
    ---------
    item_id: str
        the ID of the item to be saved

    data: dict
        the data to be saved

    save_path: path-like
        the destination path
    """
    file_path = save_path / f"{item_id}.pt"
    torch.save(data, file_path)


def save_npy(item_id, data, save_path):
    """Saves the data in numpy format (one file per sample per feature)

    Arguments
    ---------
    item_id: str
        the ID of the item to be saved

    data: dict
        the data to be saved

    save_path: path-like
        the destination path
    """
    for key, value in data.items():
        file_path = save_path / f"{key}_{item_id}.npy"
        np.save(file_path, value)


def load_pt(save_path, item_id, features):
    file_path = save_path / f"{item_id}.pt"
    return torch.load(file_path)


def load_npy(save_path, item_id, features):
    return {
        key: np.load(save_path / f"{key}_{item_id}.npy")
        for key in features
    }


SAVE_FORMATS = {
    "pt": save_pt,
    "npy": save_npy,
}

LOAD_FORMATS = {
    "pt": load_pt,
    "npy": load_npy,
}


def add_prepared_features(
    dataset,
    save_path,
    features,
    id_key="id",
    save_format="npy"
):
    load_fn = LOAD_FORMATS.get(save_format, save_format)
    save_path = Path(save_path)

    @sb.utils.data_pipeline.takes(id_key)
    @sb.utils.data_pipeline.provides(*features)
    def prepared_features_pipeline(id_key):
        data = load_fn(save_path, id_key, features)
        for feature in features:
            yield data[feature]

    dataset.add_dynamic_item(prepared_features_pipeline)
