from typing import Literal, Optional

import albumentations as A
import fiftyone as fo
import hydra
import numpy as np
import omegaconf
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT

from lightningsparseinst.utils.transforms import ResizeShortestEdge


class SegmentationDataset(Dataset):
    """A custom dataset class for segmentation tasks using the FiftyOne library.

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

    def __init__(
        self,
        ref: fo.Dataset | str,
        name: Optional[str] = None,
        split: str = "train",
        gt_field: Optional[str] = "ground_truth",
        detection_field: Optional[Literal["detections", "polylines", "segmentations"]] = "detections",
        max_num_instances_per_image: Optional[int] = None,
        transform: Optional[A.Compose] = None,
    ):
        if isinstance(ref, str):
            s = fo.load_dataset(ref)
        else:
            s = ref
        self.name = name

        self.samples = s.match_tags(split)

        self.gt_field = gt_field
        # Ensure that if the detection field is detections, it has a mask field
        self.detection_field = detection_field
        if self.detection_field == "detections" and not self.samples.has_field(f"{self.gt_field}.detections.mask"):
            raise ValueError("Detections field must have a mask field, need masks to perform segmentation")
        # Getting classes
        self.classes: list = list(self.samples.distinct(f"{self.gt_field}.{self.detection_field}.label"))
        self.img_paths = self.samples.values("filepath")
        self.labels_map_rev = {cls: idx for idx, cls in enumerate(self.classes)}

        self.max_num_instances_per_image = max_num_instances_per_image
        if not self.max_num_instances_per_image:
            self.max_num_instances_per_image = len(self.classes)

        if split == "train" and transform is not None:
            self.transform: A.Compose = transform
        else:
            self.transform: A.Compose = A.Compose([A.Compose([ResizeShortestEdge(640, 853)]), ToTensorV2()])

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size
        img = np.asarray(img)

        if not sample[self.gt_field]:
            transformed = self.transform(image=np.asarray(img))

            return transformed["image"], torch.tensor([]), torch.tensor([])

        ground_truth = sample[self.gt_field]
        if self.detection_field == "detections":
            ground_truth = ground_truth.to_polylines()

        masks = []
        labels = []
        for polyline in ground_truth.polylines:
            # The output of a segmentation conversion here is 0 for false and 255 for true. We want it to be 1 for true
            arr = polyline.to_segmentation(frame_size=img_size, target=1).mask
            masks.append(arr)
            labels.append(self.labels_map_rev[polyline.label])
        if len(masks) < len(self.classes):
            masks += [np.zeros(img_size, dtype=np.uint8)] * (len(self.classes) - len(masks))
            labels += [self.labels_map_rev[polyline.label]] * (len(self.classes) - len(masks))
        #
        # else:
        #     # we assume that mask property is present for detections
        #     # masks = [detection.mask.astype(np.uint8) for detection in sample[self.gt_field].detections]
        #     # labels = [self.labels_map_rev[detection.label] for detection in sample[self.gt_field].detections]
        #     for detection in sample[self.gt_field].detections:
        #         arr = detection.to_segmentation(frame_size=img_size).mask
        #         arr = (arr == 255).astype(np.uint8)
        #         masks.append(arr)
        #         labels.append(self.labels_map_rev[detection.label])

        transformed = self.transform(image=img, masks=masks)
        return (transformed["image"], torch.stack(transformed["masks"]), torch.tensor(labels))

    def __len__(self):
        return len(self.img_paths)

    def get_label_name_from_id(self, id: int) -> str:
        return self.classes[id]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    ds: SegmentationDataset = hydra.utils.instantiate(cfg.nn.data.dataset)

    var = ds[0]  # noqa


if __name__ == "__main__":
    main()
