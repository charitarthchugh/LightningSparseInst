from pathlib import Path

import albumentations as A
import cv2
import hydra
import lmdb
import numpy as np
import omegaconf
import orjson
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import trange

from nn_core.common import PROJECT_ROOT


class SegmentationDataset(Dataset):
    def __init__(self, lmdb_cache_dir: Path, transform: A.Compose = None, max_detections: int = None):
        self.lmdb_cache_dir = lmdb_cache_dir
        self.transform = transform or A.Compose([ToTensorV2()])
        self.max_detections = max_detections
        if not self.max_detections:
            raise ValueError("max_detections must be specified!")

        self.lmdb_conn = lmdb.Environment(
            str(lmdb_cache_dir), readonly=True, lock=False, map_async=True, map_size=8589934592
        )

        with self.lmdb_conn.begin(write=False) as txn:
            metadata = txn.get("metadata".encode())
            if metadata is None:
                raise ValueError("No metadata found in the LMDB cache!")
            metadata = orjson.loads(metadata)
            self.label_map_rv = metadata["label_map_rv"]
            self.classes = self.label_map_rv.keys()
            self.num_classes = len(self.classes)
            self.num_samples = metadata["num_samples"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with self.lmdb_conn.begin(write=False) as txn:
            img_data = txn.get(str(idx).encode())
            img_data = orjson.loads(img_data)

        img = cv2.imread(img_data["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = img.shape[:2]
        masks, labels = self.extract_masks_and_labels(img_data, height, width)

        transformed = self.transform(image=img, masks=masks)

        return transformed["image"], transformed["masks"], torch.tensor(labels)

    def extract_masks_and_labels(self, img_data, height, width):
        masks = []
        labels = []
        for annot_dict in img_data["annotations"]:
            labels.append(annot_dict["category_id"])
            polys, tmp_masks = self.process_polygons(annot_dict["segmentation"], width, height)
            combined_mask = self.combine_masks(tmp_masks, height, width)
            masks.append(combined_mask)
        return masks, labels

    def process_polygons(self, segmentation, width, height):
        polys = []
        tmp_masks = []
        for indiv_poly in segmentation:
            poly = np.array(indiv_poly, dtype=np.float32)
            poly[:, 0] *= width  # multiply x-coordinates by width
            poly[:, 1] *= height  # multiply y-coordinates by height
            poly = poly.round().astype(np.int32)
            polys.append(poly)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], color=1)  # Directly fill the mask; no need to use polys list here
            tmp_masks.append(mask)
        return polys, tmp_masks

    def combine_masks(self, tmp_masks, height, width):
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for mask in tmp_masks:
            combined_mask = np.maximum(combined_mask, mask)
        return combined_mask


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    from nn_core.common import PROJECT_ROOT

    ds: SegmentationDataset = SegmentationDataset(
        PROJECT_ROOT / ".cache" / "data" / "coco-2017" / "train", max_detections=100
    )

    for i in trange(len(ds)):
        img, masks, labels = ds[i]


if __name__ == "__main__":
    main()
