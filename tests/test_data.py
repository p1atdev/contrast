import torch

from contrast.data.cifar import TensorAugment


def test_tensor_augmentation_is_generator_deterministic() -> None:
    transform = TensorAugment(4, 0.5, 0.5, 0.2)
    image = torch.arange(3 * 32 * 32, dtype=torch.int64).reshape(3, 32, 32).to(torch.uint8)
    first = transform(image.clone(), torch.Generator().manual_seed(42))
    second = transform(image.clone(), torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second)
