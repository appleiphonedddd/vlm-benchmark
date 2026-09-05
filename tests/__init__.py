import sys
import types
from unittest.mock import MagicMock

# 1. PIL
if "PIL" not in sys.modules:
    pil = types.ModuleType("PIL")
    pil.Image = MagicMock()
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = pil.Image

# 2. tqdm
if "tqdm" not in sys.modules or not callable(getattr(sys.modules["tqdm"], "tqdm", None)):
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda x, **kwargs: x
    tqdm_mod.write = lambda x: None
    sys.modules["tqdm"] = tqdm_mod

# 3. torch
if "torch" not in sys.modules:
    torch_mod = MagicMock()
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    sys.modules["torch"] = torch_mod

# 4. transformers
if "transformers" not in sys.modules:
    tr = types.ModuleType("transformers")
    tr.__path__ = []
    tr.AutoProcessor = MagicMock()
    tr.AutoModelForImageTextToText = MagicMock()
    tr.cache_utils = MagicMock()
    tr.masking_utils = MagicMock()
    tr.modeling_outputs = MagicMock()
    sys.modules["transformers"] = tr
    sys.modules["transformers.cache_utils"] = tr.cache_utils
    sys.modules["transformers.masking_utils"] = tr.masking_utils
    sys.modules["transformers.modeling_outputs"] = tr.modeling_outputs

# 5. datasets
if "datasets" not in sys.modules:
    datasets_mod = types.ModuleType("datasets")
    datasets_mod.load_dataset = MagicMock()
    sys.modules["datasets"] = datasets_mod

