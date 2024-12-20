import logging
from functools import cached_property
from pathlib import Path
from typing import List, Mapping, Optional

import fiftyone as fo
import hydra
import lightning as L
import lmdb
import omegaconf
import torch
from albumentations import Compose
from fiftyone import ViewField
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT

from lightningsparseinst.data.dataset import SegmentationDataset
from lightningsparseinst.utils.fiftyone_io import LMDBDetectionDatasetExporter

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, class_vocab: Mapping[str, int]):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            class_vocab: association between class names and their indices
        """
        # example
        self.class_vocab: Mapping[str, int] = class_vocab

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        # example
        (dst_path / "class_vocab.tsv").write_text(
            "\n".join(f"{key}\t{value}" for key, value in self.class_vocab.items())
        )

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        # example
        lines = (src_path / "class_vocab.tsv").read_text(encoding="utf-8").splitlines()

        class_vocab = {}
        for line in lines:
            key, value = line.strip().split("\t")
            class_vocab[key] = value

        return MetaData(class_vocab=class_vocab)

    def __repr__(self) -> str:
        attributes = ",\n    ".join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}(\n    {attributes}\n)"


def collate_fn(batch):
    return tuple(zip(*batch))


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset: SegmentationDataset = worker_info.dataset
    dataset.lmdb_conn = lmdb.open(str(dataset.lmdb_cache_dir), readonly=True, lock=False, max_readers=32)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        accelerator: str,
        cache_dir: str,
        # example
    ):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = accelerator is not None and str(accelerator) == "gpu"

        self.fiftyone_dataset: fo.Dataset = None
        self.classes: List[str] | str = None
        self.split_names: Mapping[str, str] = None

        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.transform: Compose = None

        cache_dir = Path(cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir

        cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = cache_dir

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        return MetaData(class_vocab=self.train_dataset.label_map_rv)

    def prepare_data(self) -> None:
        # download only\
        pass

    def setup(self, stage: Optional[str] = None):
        self.fiftyone_dataset = fo.load_dataset(self.dataset.ref)
        self.fiftyone_dataset.compute_metadata()

        self.split_names = self.dataset.split_names

        self.transform = hydra.utils.instantiate(self.dataset.transform)
        # Label filtering logic
        self.classes = self.dataset.classes if "classes" in self.dataset.keys() else None
        if self.classes:
            if isinstance(self.classes, list):
                self.fiftyone_dataset = self.fiftyone_dataset.filter_labels(
                    f"{self.dataset.gt_field}.{self.dataset.detection_field}", ViewField("label").is_in(self.classes)
                )
            elif isinstance(self.classes, str):
                # regex case
                self.fiftyone_dataset = self.fiftyone_dataset.filter_labels(
                    f"{self.dataset.gt_field}.{self.dataset.detection_field}", ViewField("label").re_match(self.classes)
                )
        else:
            self.classes = self.fiftyone_dataset.distinct(
                f"{self.dataset.gt_field}.{self.dataset.detection_field}.label"
            )

        label_map_rv = {cls: idx for idx, cls in enumerate(self.classes)}
        dataset_cache = self.cache_dir / self.dataset.ref
        dataset_cache.mkdir(exist_ok=True, parents=True)

        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_dataset is None):
            self._handle_lmdb_caching(dataset_cache, self.split_names["train"], label_map_rv)
            self.train_dataset = SegmentationDataset(
                dataset_cache / self.split_names["train"],
                transform=self.transform,
                max_detections=self.dataset.max_detections,
            )
            self._handle_lmdb_caching(dataset_cache, self.split_names["validation"], label_map_rv)
            self.val_dataset = SegmentationDataset(
                dataset_cache / self.split_names["validation"], max_detections=self.dataset.max_detections
            )
        #
        if stage is None or stage == "test":
            self._handle_lmdb_caching(dataset_cache, self.split_names["test"], label_map_rv)
            self.test_dataset = SegmentationDataset(
                dataset_cache / self.split_names["test"], max_detections=self.dataset.max_detections
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=16,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            prefetch_factor=16,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            prefetch_factor=16,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.dataset=}, " f"{self.num_workers=}, " f"{self.batch_size=})"

    def _handle_lmdb_caching(self, dataset_cache_path, split_name, label_map_rv) -> None:
        split_cache_dir = dataset_cache_path / split_name

        mdb_files = list(split_cache_dir.glob("*.mdb")) if split_cache_dir.exists() else None
        if mdb_files:
            pylogger.info(f"Found lmdb files in {split_name} cache directory: {mdb_files}")
        else:
            pylogger.info(f"Exporting lmdb files to {split_name} cache directory: {split_cache_dir}")
            dataset_split = self.fiftyone_dataset.match_tags(split_name)
            dataset_split.export(
                dataset_exporter=LMDBDetectionDatasetExporter(
                    export_dir=split_cache_dir, gt_field=self.dataset.gt_field, label_map_rv=label_map_rv
                )
            )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    m: L.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    m.metadata
    m.setup()

    for idx, _ in enumerate(tqdm(m.train_dataloader())):
        pass


if __name__ == "__main__":
    main()
