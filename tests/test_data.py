from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import torch
from PIL import Image

from contrast.data import cifar
from contrast.data.cifar import TensorAugment, _stratified_split


def test_tensor_augmentation_is_generator_deterministic() -> None:
    transform = TensorAugment(4, 0.5, 0.5, 0.2)
    image = torch.arange(3 * 32 * 32, dtype=torch.int64).reshape(3, 32, 32).to(torch.uint8)
    first = transform(image.clone(), torch.Generator().manual_seed(42))
    second = transform(image.clone(), torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second)


def test_huggingface_parquet_is_converted_and_cached(tmp_path, monkeypatch) -> None:
    encoded = BytesIO()
    Image.fromarray(np.full((32, 32, 3), 17, dtype=np.uint8)).save(encoded, format="PNG")
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    table = pa.table(
        {
            "img": pa.array([{"bytes": encoded.getvalue(), "path": None}], type=image_type),
            "fine_label": pa.array([7], type=pa.int64()),
        }
    )
    parquet_path = tmp_path / "train.parquet"
    parquet.write_table(table, parquet_path)
    monkeypatch.setattr(cifar, "hf_hub_download", lambda **_: str(parquet_path))
    cifar._TENSOR_CACHE.clear()

    images, labels = cifar._load_huggingface(tmp_path, "train", True, "test/cifar", "revision")
    assert images.shape == (1, 3, 32, 32)
    assert images.unique().item() == 17
    assert labels.tolist() == [7]

    cifar._TENSOR_CACHE.clear()

    def fail_download(**_):
        raise AssertionError("processed cache should avoid another Hub request")

    monkeypatch.setattr(cifar, "hf_hub_download", fail_download)
    cached_images, cached_labels = cifar._load_huggingface(
        tmp_path,
        "train",
        False,
        "test/cifar",
        "revision",
    )
    torch.testing.assert_close(cached_images, images)
    torch.testing.assert_close(cached_labels, labels)


def test_stratified_split_is_deterministic_for_its_own_seed() -> None:
    labels = torch.arange(4).repeat_interleave(10)

    first_train, first_validation = _stratified_split(labels, 0.2, seed=13)
    second_train, second_validation = _stratified_split(labels, 0.2, seed=13)
    _, other_validation = _stratified_split(labels, 0.2, seed=14)

    assert first_train == second_train
    assert first_validation == second_validation
    assert first_validation != other_validation
    assert len(first_train) == 32
    assert len(first_validation) == 8
    assert torch.bincount(labels[first_validation], minlength=4).tolist() == [2, 2, 2, 2]
