from torchtitan.protocols.model_spec import ModelSpec

from miles.backends.torchtitan_utils.models.kimi_k3.model import KimiK3Model
from miles.backends.torchtitan_utils.models.kimi_k3.parallelize import parallelize_kimi_k3
from miles.backends.torchtitan_utils.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


def model_registry(flavor: str, attn_backend: str = "flex") -> ModelSpec:
    if flavor != "4layer-64experts":
        raise ValueError(f"Only K3's 4layer-64experts validation flavor is supported, got {flavor!r}")
    return ModelSpec(
        name="kimi_k3",
        flavor=flavor,
        model=KimiK3Model.Config(),
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiK3StateDictAdapter,
    )
