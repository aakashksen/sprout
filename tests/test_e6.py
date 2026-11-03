import numpy as np
import pytest
import torch
from PIL import Image

from sprout.preprocess import preprocess
from sprout.preprocess_np import preprocess_array


def _photo(w=97, h=61, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_numpy_preprocess_is_identical_to_the_torch_one(mode):
    image = _photo().convert(mode)
    got = preprocess_array(image)
    assert got.dtype == np.float32 and got.shape == (3, 224, 224)
    assert torch.equal(preprocess(image), torch.from_numpy(got))
    assert torch.equal(preprocess(image, 64), torch.from_numpy(preprocess_array(image, 64)))
