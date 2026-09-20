from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class WasteDataset(Dataset):
    """
    PyTorch Dataset for the waste-classification project.

    Images are loaded according to a frozen CSV manifest.
    """

    def __init__(
        self,
        manifest_path,
        dataset_root,
        transform=None,
        class_to_idx=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        self.data = pd.read_csv(self.manifest_path)

        required_columns = {"relative_path", "class_name"}

        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Manifest is missing columns: {missing_columns}"
            )

        if class_to_idx is None:
            class_names = sorted(
                self.data["class_name"].unique()
            )

            self.class_to_idx = {
                class_name: idx
                for idx, class_name in enumerate(class_names)
            }
        else:
            self.class_to_idx = class_to_idx

        self.classes = [
            class_name
            for class_name, _ in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = (
            self.dataset_root / row["relative_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            # Important because our audit found:
            # RGB, RGBA, P, CMYK and L images.
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        label = self.class_to_idx[row["class_name"]]

        return image, label