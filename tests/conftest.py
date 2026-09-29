import pytest
import torch

flashinfer = pytest.importorskip("flashinfer", reason="the conformance suite needs FlashInfer")


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        pytest.skip(f"this suite targets SM12x; found SM{major}{minor}")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def fi():
    return flashinfer
