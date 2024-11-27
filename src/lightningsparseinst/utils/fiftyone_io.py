import gc
from dataclasses import dataclass

import fiftyone as fo
import fiftyone.utils.data as foud
import lmdb
import orjson


@dataclass
class CocoAnnotation:
    category_id: int
    segmentation: list
    iscrowd: int


@dataclass
class CocoImage:
    path: str
    annotations: list[CocoAnnotation]


@dataclass
class CocoMetadata:
    label_map_rv: dict[str, int]
    num_samples: int


class LMDBDetectionDatasetExporter(foud.LabeledImageDatasetExporter):
    """Exporter for detecting datasets to LMDB format.

    Datasets of this type are exported with the images and
    annotations stored in an LMDB database.

    Args:
        export_dir: the directory to write the export
        lmdb_path: where to save the lmdb files
        gt_field: ground truth field specifying labels for the dataset
        detection_field: field that specifies the type of detection, default is "detections"
    """

    def __init__(self, export_dir, gt_field, label_map_rv):
        super().__init__(export_dir=export_dir)
        self.lmdb_path = export_dir
        self.gt_field = gt_field
        self.conn = None
        self.label_map_rv = label_map_rv
        self.num_classes = 0
        self.id = 0

    @property
    def requires_image_metadata(self):
        return True

    @property
    def label_cls(self):
        return fo.Detections, fo.Polylines

    def setup(self):
        self.conn = lmdb.Environment(str(self.lmdb_path), map_size=1099511627776 * 2)

    def export_sample(self, image_path: str, label: fo.Detections | fo.Polylines, metadata: fo.ImageMetadata = None):
        if not label:
            coco_img = CocoImage(image_path, [])
            json_data = orjson.dumps(coco_img)
            with self.conn.begin(write=True) as txn:
                txn.put(str(self.id).encode(), json_data)
            self.id += 1
            return
        elif isinstance(label, fo.Detections):
            annots = label.detections
        else:
            annots = label.polylines
        coco_annots = []
        for annot in annots:
            if annot.label in self.label_map_rv.keys():
                category_id = self.label_map_rv[annot.label]
            else:
                self.label_map_rv[annot.label] = self.num_classes
                category_id = self.num_classes
                self.num_classes += 1
            # segmentation = COCOObject.from_label(annot, metadata=metadata, category_id=category_id).to_polyline(
            #     frame_size=(metadata.width, metadata.height)
            # )
            if isinstance(annot, fo.Detection):
                annot = annot.to_polyline()

            coco_annots.append(
                CocoAnnotation(
                    category_id=category_id,
                    segmentation=annot.points,
                    iscrowd=label.iscrowd if hasattr(label, "iscrowd") else 0,
                )
            )

        coco_img = CocoImage(path=image_path, annotations=coco_annots)
        json_data = orjson.dumps(coco_img)
        with self.conn.begin(write=True) as txn:
            txn.put(str(self.id).encode(), json_data)
        self.id += 1

    def close(self, *args):
        if self.conn:
            metadata = CocoMetadata(label_map_rv=self.label_map_rv, num_samples=self.id)
            with self.conn.begin(write=True) as txn:
                # txn.put("label_map_rev".encode(), orjson.dumps(self.label_map_rev))
                # txn.put("classes".encode(), orjson.dumps(list(self.label_map_rev.keys())))
                # txn.put("".encode(), str(self.num_classes).encode())
                txn.put("metadata".encode(), orjson.dumps(metadata))
            self.conn.close()
        gc.collect()


# Example Usage
# dataset = fo.load_dataset("my-dataset")
# exporter = LMDBDetectionDatasetExporter("/path/to/export", "/path/to/lmdb", "ground_truth")
# dataset.export(exporter)
