"""W6: check of ImageDataset (moved from W5 on 5.11.2026).

Run from the project root:
    uv run pytest exercises/w06
"""

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from mlproject.data.dataset import ImageDataset, read_split


def test_image_dataset(tmp_path):
    items = []
    for i, (mode, color) in enumerate([("RGB", (255, 0, 0)), ("L", 128), ("RGBA", (0, 0, 255, 255))]):
        path = tmp_path / f"img_{i}.png"
        Image.new(mode, (20, 10), color).save(path)
        items.append((path, i))
    split = tmp_path / "split.csv"
    split.write_text("path,label\n" + "".join(f"{p},{label}\n" for p, label in items))
    ds = ImageDataset(read_split(split), transform=transforms.ToTensor())
    assert len(ds) == 3
    image, label = ds[1]
    assert label == 1 and image.shape == (3, 10, 20)  # a grayscale file comes back as RGB
    x, y = next(iter(DataLoader(ds, batch_size=3)))
    assert x.shape == (3, 3, 10, 20) and y.tolist() == [0, 1, 2]
