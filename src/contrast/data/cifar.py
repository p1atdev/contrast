from __future__ import annotations

import hashlib
import pickle
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from contrast.config.schema import ExperimentConfig
from contrast.runtime.context import RuntimeContext

_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
_CIFAR_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
_MEAN = torch.tensor((0.5071, 0.4867, 0.4408)).view(3, 1, 1)
_STD = torch.tensor((0.2675, 0.2565, 0.2761)).view(3, 1, 1)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_and_extract(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-100-python.tar.gz"
    if not archive.exists() or _md5(archive) != _CIFAR_MD5:
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(_CIFAR_URL, temporary)
        if _md5(temporary) != _CIFAR_MD5:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("CIFAR-100 archive checksum mismatch")
        temporary.replace(archive)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root, filter="data")


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
    ) -> None:
        directory = root / "cifar-100-python"
        if not directory.exists():
            if not download:
                raise FileNotFoundError(f"CIFAR-100 was not found under {root}")
            _download_and_extract(root)
        with (directory / split).open("rb") as stream:
            payload: dict[bytes, Any] = pickle.load(stream, encoding="bytes")
        self.images = torch.from_numpy(
            np.asarray(payload[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        )
        self.labels = torch.tensor(payload[b"fine_labels"], dtype=torch.long)
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
            generator.manual_seed(self.seed + epoch * len(self) * self.views + index * self.views + view)
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


def _stratified_split(labels: torch.Tensor, fraction: float, seed: int) -> tuple[list[int], list[int]]:
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
    )
    evaluation_train = Cifar100Dataset(
        config.data.root,
        "train",
        1,
        config.run.seed,
        None,
        False,
    )
    test_dataset = Cifar100Dataset(
        config.data.root,
        "test",
        1,
        config.run.seed,
        None,
        config.data.download,
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
