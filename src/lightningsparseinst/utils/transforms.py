import albumentations as A
import cv2
import numpy as np


class ResizeShortestEdge(A.DualTransform):
    """ResizeShortestEdge transformation class.

    This class provides functionality to resize an image such that its shortest
    edge matches a specified size while maintaining the aspect ratio. The
    longest edge is also constrained by a maximum size to avoid excessively
    large dimensions.

    Attributes:
        shortest_max_size (list[int] | int): Target size(s) for the shortest edge of the image.
        largest_max_size (int): Maximum allowable size for the longest edge of the image.
        interpolation: OpenCV interpolation method.
    """

    def __init__(
        self,
        shortest_max_size: list[int] | int,
        largest_max_size: int,
        interpolation=cv2.INTER_LINEAR,
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply, p)

        if isinstance(shortest_max_size, int):
            self.shortest_max_size = [shortest_max_size]
        else:
            self.shortest_max_size = shortest_max_size

        self.largest_max_size = largest_max_size
        self.interpolation = interpolation

    def apply(self, img, shortest_max_size=0, largest_max_size=0, **params):
        h, w = img.shape[:2]

        # Compute new size
        scale = shortest_max_size / min(h, w)
        new_h, new_w = (shortest_max_size, int(scale * w)) if h < w else (int(scale * h), shortest_max_size)

        # Check if the largest edge is larger than the target largest_max_size
        if max(new_h, new_w) > largest_max_size:
            scale = largest_max_size / max(new_h, new_w)
            new_h = max(1, int(new_h * scale))
            new_w = max(1, int(new_w * scale))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

        return img_resized

    def get_params(self):
        min_size = np.random.choice(self.shortest_max_size)
        max_size = self.largest_max_size
        return {"shortest_max_size": min_size, "largest_max_size": max_size}

    def get_transform_init_args_names(self):
        return ("shortest_max_size", "largest_max_size", "interpolation")


if __name__ == "__main__":
    import fiftyone as fo
    from PIL import Image

    ds = fo.load_dataset("coco-2017")
    img_pth = ds.values("filepath")[0]
    sample = ds[img_pth]
    img = Image.open(img_pth).convert("RGB")
    img.show()
    img = np.asarray(img)
    transforms = A.Compose([ResizeShortestEdge([416, 448, 480, 512, 544, 576, 608, 640], 853)])
    transformed = transforms(image=img)["image"]

    transformed_img = Image.fromarray(transformed)
    transformed_img.show()
