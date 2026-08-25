from __future__ import annotations

import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from contrast.config.schema import ExperimentConfig
from contrast.runtime.context import RuntimeContext

_HF_FILES = {
    "train": "cifar100/train-00000-of-00001.parquet",
    "test": "cifar100/test-00000-of-00001.parquet",
}
_MEAN = torch.tensor((0.5071, 0.4867, 0.4408)).view(3, 1, 1)
_STD = torch.tensor((0.2675, 0.2565, 0.2761)).view(3, 1, 1)
_TENSOR_CACHE: dict[tuple[Path, str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _load_huggingface(
    root: Path,
    split: str,
    download: bool,
    repo: str,
    revision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split not in _HF_FILES:
        raise ValueError(f"unsupported CIFAR-100 split: {split}")
    cache_key = (root.resolve(), repo, revision, split)
    if cache_key in _TENSOR_CACHE:
        return _TENSOR_CACHE[cache_key]

    source_key = f"{repo.replace('/', '--')}--{revision[:12]}"
    converted = root / "processed" / source_key / f"{split}.pt"
    if converted.exists():
        payload = torch.load(converted, map_location="cpu", weights_only=True)
        result = payload["images"], payload["labels"]
        _TENSOR_CACHE[cache_key] = result
        return result

    root.mkdir(parents=True, exist_ok=True)
    parquet_path = hf_hub_download(
        repo_id=repo,
        filename=_HF_FILES[split],
        repo_type="dataset",
        revision=revision,
        cache_dir=root / ".hf-cache",
        local_files_only=not download,
    )
    table = parquet.read_table(parquet_path, columns=["img", "fine_label"])
    records = table["img"].to_pylist()
    images = torch.empty((len(records), 3, 32, 32), dtype=torch.uint8)
    for index, record in enumerate(records):
        encoded = record["bytes"]
        if encoded is None:
            raise ValueError("Hugging Face CIFAR-100 row did not contain encoded image bytes")
        with Image.open(BytesIO(encoded)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        images[index].copy_(torch.from_numpy(array.copy()).permute(2, 0, 1))
    labels = torch.from_numpy(
        np.asarray(table["fine_label"].to_numpy(zero_copy_only=False), dtype=np.int64).copy()
    )

    converted.parent.mkdir(parents=True, exist_ok=True)
    temporary = converted.with_suffix(".tmp")
    torch.save({"images": images, "labels": labels}, temporary)
    temporary.replace(converted)
    result = images, labels
    _TENSOR_CACHE[cache_key] = result
    return result


class TensorAugment(nn.Module):
    def __init__(
        self,
        crop_padding: int,
        flip_probability: float,
        color_jitter_strength: float,
        grayscale_probability: float,
    ) -> None:
        super().__init__()
        self.crop_padding = crop_padding
        self.flip_probability = flip_probability
        self.jitter = color_jitter_strength
        self.grayscale_probability = grayscale_probability

    @staticmethod
    def _uniform(generator: torch.Generator, low: float, high: float) -> float:
        return float(torch.empty((), dtype=torch.float32).uniform_(low, high, generator=generator))

    def forward(self, image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        image = image.float().div_(255.0)
        if self.crop_padding:
            padded = F.pad(image, (self.crop_padding,) * 4, mode="reflect")
            limit = 2 * self.crop_padding + 1
            top = int(torch.randint(limit, (), generator=generator))
            left = int(torch.randint(limit, (), generator=generator))
            image = padded[:, top : top + 32, left : left + 32]
        if float(torch.rand((), generator=generator)) < self.flip_probability:
            image = image.flip(-1)
        if self.jitter:
            brightness = self._uniform(generator, 1.0 - 0.8 * self.jitter, 1.0 + 0.8 * self.jitter)
            contrast = self._uniform(generator, 1.0 - 0.8 * self.jitter, 1.0 + 0.8 * self.jitter)
            saturation = self._uniform(generator, 1.0 - 0.8 * self.jitter, 1.0 + 0.8 * self.jitter)
            order = torch.randperm(3, generator=generator).tolist()
            for operation in order:
                if operation == 0:
                    image = image * brightness
                elif operation == 1:
                    image = (image - image.mean((1, 2), keepdim=True)) * contrast + image.mean(
                        (1, 2), keepdim=True
                    )
                else:
                    gray = image.mean(0, keepdim=True)
                    image = (image - gray) * saturation + gray
            image = image.clamp_(0.0, 1.0)
        if float(torch.rand((), generator=generator)) < self.grayscale_probability:
            image = image.mean(0, keepdim=True).expand_as(image)
        return (image - _MEAN) / _STD


class Cifar100Dataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(
        self,
        root: Path,
        split: str,
        views: int,
        seed: int,
        augment: TensorAugment | None,
        download: bool,
        huggingface_repo: str,
        huggingface_revision: str,
    ) -> None:
        self.images, self.labels = _load_huggingface(
            root,
            split,
            download,
            huggingface_repo,
            huggingface_revision,
        )
        self.views = views
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, item: int | tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        epoch, index = item if isinstance(item, tuple) else (0, item)
        image = self.images[index]
        rendered = []
        for view in range(self.views):
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + epoch * len(self) * self.views + index * self.views + view
            )
            if self.augment is None:
                rendered.append((image.float() / 255.0 - _MEAN) / _STD)
            else:
                rendered.append(self.augment(image.clone(), generator))
        return torch.stack(rendered), int(self.labels[index]), index


class EpochBatchSampler(Sampler[list[tuple[int, int]]]):
    def __init__(
        self,
        indices: list[int],
        global_batch_size: int,
        seed: int,
        runtime: RuntimeContext,
        drop_last: bool = True,
    ) -> None:
        if global_batch_size % runtime.world_size:
            raise ValueError("global source batch size must be divisible by world size")
        self.indices = indices
        self.local_batch_size = global_batch_size // runtime.world_size
        self.seed = seed
        self.runtime = runtime
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        global_order = [self.indices[position] for position in order]
        local_order = global_order[self.runtime.rank :: self.runtime.world_size]
        for start in range(0, len(local_order), self.local_batch_size):
            batch = local_order[start : start + self.local_batch_size]
            if len(batch) == self.local_batch_size or not self.drop_last:
                yield [(self.epoch, index) for index in batch]

    def __len__(self) -> int:
        size = len(self.indices) // self.runtime.world_size
        if self.drop_last:
            return size // self.local_batch_size
        return (size + self.local_batch_size - 1) // self.local_batch_size


def _stratified_split(
    labels: torch.Tensor, fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not fraction:
        return list(range(labels.numel())), []
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_class.setdefault(label, []).append(index)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)
        count = max(1, round(len(indices) * fraction))
        validation_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    return sorted(train_indices), sorted(validation_indices)


@dataclass
class DataBundle:
    train: DataLoader
    validation: DataLoader | None
    test: DataLoader
    memory: DataLoader
    train_sampler: EpochBatchSampler


def build_cifar100_loaders(config: ExperimentConfig, runtime: RuntimeContext) -> DataBundle:
    augmentation = TensorAugment(
        config.data.augmentation.crop_padding,
        config.data.augmentation.horizontal_flip_probability,
        config.data.augmentation.color_jitter_strength,
        config.data.augmentation.grayscale_probability,
    )
    train_dataset = Cifar100Dataset(
        config.data.root,
        "train",
        config.batch.views,
        config.run.seed,
        augmentation,
        config.data.download,
        config.data.huggingface.repo_id,
        config.data.huggingface.revision,
    )
    evaluation_train = Cifar100Dataset(
        config.data.root,
        "train",
        1,
        config.run.seed,
        None,
        False,
        config.data.huggingface.repo_id,
        config.data.huggingface.revision,
    )
    test_dataset = Cifar100Dataset(
        config.data.root,
        "test",
        1,
        config.run.seed,
        None,
        config.data.download,
        config.data.huggingface.repo_id,
        config.data.huggingface.revision,
    )
    train_indices, validation_indices = _stratified_split(
        train_dataset.labels,
        config.data.validation_fraction,
        config.run.seed,
    )
    batch_sampler = EpochBatchSampler(
        train_indices,
        config.batch.global_source_batch_size,
        config.run.seed,
        runtime,
    )
    loader_options = {
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": config.data.num_workers > 0,
    }
    train = DataLoader(train_dataset, batch_sampler=batch_sampler, **loader_options)
    evaluation_batch_size = min(512, config.batch.global_source_batch_size)
    validation = (
        DataLoader(
            Subset(evaluation_train, validation_indices),
            batch_size=evaluation_batch_size,
            shuffle=False,
            **loader_options,
        )
        if validation_indices
        else None
    )
    test = DataLoader(
        test_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **loader_options,
    )
    memory = DataLoader(
        Subset(evaluation_train, train_indices),
        batch_size=evaluation_batch_size,
        shuffle=False,
        **loader_options,
    )
    return DataBundle(train, validation, test, memory, batch_sampler)
