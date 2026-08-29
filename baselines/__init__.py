BASELINE_REGISTRY = {}

def build_baseline(baseline_name: str, **kwargs):
    baseline_cls = BASELINE_REGISTRY.get(baseline_name.lower())
    if baseline_cls is None:
        raise ValueError(f"Unsupported baseline: {baseline_name}. Supported baselines: {list(BASELINE_REGISTRY.keys())}")
    return baseline_cls(**kwargs)
