import re
from typing import Literal, Optional

import albumentations as A
import fiftyone as fo
import hydra
import numpy as np
import omegaconf
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from fiftyone import ViewField
from nn_core.common import PROJECT_ROOT
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    A custom dataset class for segmentation tasks using the FiftyOne library.

    Args:
        fiftyone_dataset (fo.Dataset | str): The FiftyOne dataset or name of the dataset to load.
        split (Optional[str]): The split tag to filter samples. If provided, only samples with this tag will be used.
        classes (Optional[list[str] | str]): List of class labels or a regex string to filter class labels.
        gt_field (str): The ground truth field in the dataset. Default is "ground_truth".
        detection_field (Optional[Literal['detections', 'polylines', 'segmentations']]): The field type of detections in the dataset. Default is 'detections'.
        transform (Optional[A.Compose]): The transformation to apply to dataset items. If None, applies default transformation with ToTensorV2.

    Raises:
        ValueError: If `split` is None and dataset has more than one tag, or if required fields are missing in the dataset.

    Attributes:
        samples (fo.DatasetView): The filtered dataset according to the specified split and classes.
        gt_field (str): The ground truth field.
        detection_field (str): The type of detection field.
        classes (list[str]): The list of class labels.
        img_paths (list[str]): List of image file paths.
        labels_map_rev (dict): A dictionary mapping class labels to their indices.
        transform (A.Compose): The transformation to apply to dataset items.
    """
    def __init__(self, fiftyone_dataset: fo.Dataset | str, split: Optional[str] = None,
                 classes: Optional[list[str] | str] = None,
                 gt_field="ground_truth",
                 detection_field: Optional[Literal['detections', 'polylines', 'segmentations']] = 'detections',
                 transform: Optional[A.Compose] = None
                 ):

        if isinstance(fiftyone_dataset, str):
            s = fo.load_dataset(fiftyone_dataset)
        else:
            s = fiftyone_dataset

        if len(s.tags) <= 1:
            self.samples = s
        elif split is not None:
            self.samples = s.match_tags(split)
        else:
            raise ValueError("Either split or tags must be specified.")

        self.gt_field = gt_field
        # Ensure that if the detection field is detections, it has a mask field
        self.detection_field = detection_field
        if self.detection_field == "detections" and not self.samples.has_field(f"{self.gt_field}.detections.mask"):
            raise ValueError("Detections field must have a mask field, need masks to perform segmentation")
        elif self.detection_field == 'segmentations' and not self.samples.has_field(
            f"{self.gt_field}.segmentations.labels"):
            raise ValueError("Segmentations field must have a labels field, need labels to perform segmentation")
        # Getting classes
        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(f"{self.gt_field}.{self.detection_field}.label")
        # needs to be tested... implemented because of the our of roi class in cityscapes
        elif isinstance(self.classes, list):
            self.samples = set(self.samples.filter_labels(f"{self.gt_field}.{self.detection_field}",
                                                          ViewField('label').is_in(self.classes)))
        elif isinstance(self.classes, str):
            regex = re.compile(self.classes)
            self.classes = list(
                filter(regex.search, self.samples.distinct(f"{self.gt_field}.{self.detection_field}.label")))
            self.samples = self.samples.filter_labels(f"{self.gt_field}.{self.detection_field}",
                                                      ViewField('label').is_in(self.classes))
        self.img_paths = self.samples.values("filepath")
        self.labels_map_rev = {cls: idx for idx, cls in enumerate(self.classes)}

        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([ToTensorV2()])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = Image.open(img_path).convert("RGB")

        masks = []
        labels = []
        if self.detection_field == "polylines":
            for polyline in sample[self.gt_field].polylines:
                # The output of a segmentation conversion here is 0 for false and 255 for true. We want it to be 1 for true
                arr = polyline.to_segmentation(frame_size=img.size).mask
                # when the arr is 255, it is true, else false, so this becomes a bool array that we cast into uint8
                arr = (arr == 255).astype(
                    np.uint8)  # Theoretically, we can just divide the array by 255 but this should be faster.
                masks.append(arr)
                labels.append(self.labels_map_rev[polyline.label])

        else:
            # we assume that mask property is present for detections
            # masks = [detection.mask.astype(np.uint8) for detection in sample[self.gt_field].detections]
            # labels = [self.labels_map_rev[detection.label] for detection in sample[self.gt_field].detections]
            for detection in sample[self.gt_field].detections:
                arr = detection.to_segmentation(frame_size=img.size).mask
                arr = (arr == 255).astype(np.uint8)
                masks.append(arr)
                labels.append(self.labels_map_rev[detection.label])

        transformed = self.transform(image=np.asarray(img), masks=masks)
        return transformed['image'], torch.stack(transformed['masks']), torch.tensor(labels)


    def __len__(self):
        return len(self.img_paths)

    def get_label_name_from_id(self, id: int) -> str:
        return self.classes[id]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(cfg.nn.data.dataset, _recursive_=False)


if __name__ == "__main__":
    main()
