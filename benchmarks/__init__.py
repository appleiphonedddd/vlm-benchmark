from .MMBench import MMBenchDataset
from .MMMU import MMMUDataset
from .MMMUPro import MMMUProDataset

DATASET_REGISTRY = {
    "mmbench": MMBenchDataset,
    "mmmu": MMMUDataset,
    "mmmupro": MMMUProDataset,
}

def build_dataset(dataset_name: str, **kwargs):
    dataset_cls = DATASET_REGISTRY.get(dataset_name.lower())
    if dataset_cls is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_REGISTRY.keys())}")
    return dataset_cls(**kwargs)
