"""Tests for diffusion `update_weights_from_tensor`.

This module verifies in-place weight updates from serialized in-memory tensors
without restarting the server.

Test goals:
- API contract and request validation for `/update_weights_from_tensor`
- module-targeted updates (`target_modules`) for transformer/vae
- failure reporting on corrupted tensor update
- flattened-bucket payload compatibility
- offload compatibility (`--dit-layerwise-offload`)

Model-pair selection in CI follows `test_update_weights_from_disk.py`:
- local: run both pairs
- CI: run one pair (random unless `SGLANG_MMGEN_UPDATE_WEIGHTS_PAIR` is set)
"""

from __future__ import annotations

import os
import random
import subprocess
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

import pytest
import requests
import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    compute_weights_checksum,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.loader.weights_updater import (
    load_weights_into_model,
)
from sglang.multimodal_gen.runtime.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_FLUX_2_KLEIN_4B_MODEL_NAME_FOR_TEST,
    DEFAULT_FLUX_2_KLEIN_BASE_4B_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_2512_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
    get_dynamic_server_port,
    is_in_ci,
)
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

logger = init_logger(__name__)

_TRANSFORMER_MODULE = "transformer"
_VAE_MODULE = "vae"

_DIFFERING_MODULES: list[str] = [_TRANSFORMER_MODULE, _VAE_MODULE]

_ALL_MODEL_PAIRS: list[tuple[str, str]] = [
    (
        DEFAULT_FLUX_2_KLEIN_BASE_4B_MODEL_NAME_FOR_TEST,
        DEFAULT_FLUX_2_KLEIN_4B_MODEL_NAME_FOR_TEST,
    ),
    (
        DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
        DEFAULT_QWEN_IMAGE_2512_MODEL_NAME_FOR_TEST,
    ),
]

_CI_MODEL_PAIR_ENV = "SGLANG_MMGEN_UPDATE_WEIGHTS_PAIR"
_PERF_LOG_PATH = "/root/logs/perf_upwt_detailed.log"
_MULTI_GPU_PROFILING_LOG_PATH = "/root/logs/multi-gpu-profiling.log"
_TP_DEBUG_LOG_PATH = "/root/logs/tp_debug_3_24.log"
_BUCKET_SIZE_ENV = "SGLANG_MMGEN_TENSOR_BUCKET_SIZE_BYTES"
_DIT_CPU_OFFLOAD_ENV = "SGLANG_MMGEN_TEST_DIT_CPU_OFFLOAD"
_TP_SIZE_ENV = "SGLANG_TEST_TP_SIZE"
_TP_PARAM_NAME_ENV = "SGLANG_TEST_TP_PARAM_NAME"

# Ensure perf log file path exists as soon as this test module is loaded.
os.makedirs(os.path.dirname(_PERF_LOG_PATH), exist_ok=True)
with open(_PERF_LOG_PATH, "a", encoding="utf-8"):
    pass


@dataclass
class _NamedTensorUpdate:
    name: str
    tensor: torch.Tensor


def _resolve_active_model_pairs() -> list[tuple[str, str]]:
    if not is_in_ci():
        return _ALL_MODEL_PAIRS

    pair_by_id = {pair[0].split("/")[-1]: pair for pair in _ALL_MODEL_PAIRS}
    selected_pair_id = os.environ.get(_CI_MODEL_PAIR_ENV)
    if selected_pair_id is None:
        return [random.choice(_ALL_MODEL_PAIRS)]

    selected_pair = pair_by_id.get(selected_pair_id)
    if selected_pair is None:
        valid_ids = ", ".join(sorted(pair_by_id))
        raise ValueError(
            f"Invalid {_CI_MODEL_PAIR_ENV}={selected_pair_id!r}. "
            f"Expected one of: {valid_ids}."
        )
    return [selected_pair]


_ACTIVE_MODEL_PAIRS = _resolve_active_model_pairs()
_PAIR_IDS = [p[0].split("/")[-1] for p in _ACTIVE_MODEL_PAIRS]


def _get_test_tp_size() -> int:
    return int(os.environ.get(_TP_SIZE_ENV, "2"))


def _maybe_filter_tp_named_tensors(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    target_name = os.environ.get(_TP_PARAM_NAME_ENV, "").strip()
    if not target_name:
        return named_tensors
    filtered = [(name, t) for name, t in named_tensors if name == target_name]
    assert filtered, (
        f"{_TP_PARAM_NAME_ENV}={target_name!r} not found in selected params: "
        f"{[n for n, _ in named_tensors]}"
    )
    return filtered


def _pick_params_from_disk(
    model_path: str,
    module_name: str,
    num_params: int,
    max_numel: int | None = None,
    min_numel: int | None = None,
) -> list[_NamedTensorUpdate]:
    """Pick ``num_params`` parameter tensors from one module on disk.

    Preference order:
    1) floating-point parameters (so ``+delta`` is meaningful)
    2) non-floating parameters as fallback if float params are insufficient
    """
    assert num_params > 0, "num_params must be > 0"
    local_path = maybe_download_model(model_path)
    weights_dir = os.path.join(local_path, module_name)
    assert os.path.exists(
        weights_dir
    ), f"No weights dir for module '{module_name}' in {local_path}"

    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    target_name = os.environ.get(_TP_PARAM_NAME_ENV, "").strip()
    target_update: _NamedTensorUpdate | None = None
    if target_name:
        for name, tensor in safetensors_weights_iterator(safetensors_files):
            if name == target_name:
                target_update = _NamedTensorUpdate(name=name, tensor=tensor.clone())
                break
        assert target_update is not None, (
            f"{_TP_PARAM_NAME_ENV}={target_name!r} not found in module "
            f"'{module_name}' of model {model_path}"
        )
        if num_params == 1:
            return [target_update]

    picked_small_float: list[_NamedTensorUpdate] = []
    picked_small_nonfloat: list[_NamedTensorUpdate] = []
    picked_large_float: list[_NamedTensorUpdate] = []
    picked_large_nonfloat: list[_NamedTensorUpdate] = []
    for name, tensor in safetensors_weights_iterator(safetensors_files):
        if target_name and name == target_name:
            continue
        if min_numel is not None and tensor.numel() < min_numel:
            continue
        is_small = max_numel is None or tensor.numel() <= max_numel
        if tensor.is_floating_point():
            target = picked_small_float if is_small else picked_large_float
            target.append(_NamedTensorUpdate(name=name, tensor=tensor.clone()))
        else:
            target = picked_small_nonfloat if is_small else picked_large_nonfloat
            target.append(_NamedTensorUpdate(name=name, tensor=tensor.clone()))

    selected: list[_NamedTensorUpdate] = [target_update] if target_update is not None else []
    for bucket in (
        picked_small_float,
        picked_small_nonfloat,
        picked_large_float,
        picked_large_nonfloat,
    ):
        if len(selected) >= num_params:
            break
        selected.extend(bucket[: (num_params - len(selected))])

    assert selected, (
        f"No tensor found in module '{module_name}' "
        f"with constraints min_numel={min_numel}, max_numel={max_numel}"
    )
    return selected


def _pick_one_param_from_disk(model_path: str, module_name: str) -> _NamedTensorUpdate:
    return _pick_params_from_disk(model_path, module_name, num_params=1)[0]


def _pick_params_by_names_from_disk(
    model_path: str,
    module_name: str,
    param_names: list[str],
) -> list[_NamedTensorUpdate]:
    """Pick tensors by exact names and keep caller-provided order."""
    assert param_names, "param_names must be non-empty"
    local_path = maybe_download_model(model_path)
    weights_dir = os.path.join(local_path, module_name)
    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    wanted = set(param_names)
    found: dict[str, torch.Tensor] = {}
    for name, tensor in safetensors_weights_iterator(safetensors_files):
        if name in wanted:
            found[name] = tensor.clone()
            if len(found) == len(wanted):
                break
    missing = [name for name in param_names if name not in found]
    assert not missing, (
        f"Requested param(s) not found in module '{module_name}': {missing}"
    )
    return [_NamedTensorUpdate(name=name, tensor=found[name]) for name in param_names]


def _build_direct_update(
    model_path: str,
    module_name: str,
    value: float = 2.0,
    value_sequence: list[float] | None = None,
    num_params: int = 5,
    max_numel: int | None = None,
    min_numel: int | None = None,
    param_names: list[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if param_names is not None:
        picked_params = _pick_params_by_names_from_disk(
            model_path=model_path,
            module_name=module_name,
            param_names=param_names,
        )
    else:
        picked_params = _pick_params_from_disk(
            model_path=model_path,
            module_name=module_name,
            num_params=num_params,
            max_numel=max_numel,
            min_numel=min_numel,
        )
    if value_sequence is not None:
        assert len(value_sequence) >= len(picked_params), (
            f"value_sequence length ({len(value_sequence)}) must be >= "
            f"number of picked params ({len(picked_params)})"
        )
    named_tensors: list[tuple[str, torch.Tensor]] = []
    for i, picked in enumerate(picked_params):
        cur_value = value_sequence[i] if value_sequence is not None else value
        t = picked.tensor.to(device="cuda")
        if t.is_floating_point():
            updated = torch.full_like(t, cur_value)
        else:
            updated = torch.full_like(t, int(cur_value))
        named_tensors.append((picked.name, updated))
    return named_tensors


def _build_invalid_shape_update(
    model_path: str,
    module_name: str,
) -> list[tuple[str, torch.Tensor]]:
    picked = _pick_one_param_from_disk(model_path, module_name)
    t = picked.tensor.to(device="cuda")
    assert t.numel() > 1, "Need tensor with >1 elements to create invalid shape payload"

    bad = t.reshape(-1)[:-1].clone()
    return [(picked.name, bad)]


def _compute_expected_checksum_after_direct_update(
    model_path: str,
    module_name: str,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> str:
    """Compute expected module checksum after applying direct named_tensors update."""
    local_path = maybe_download_model(model_path)
    weights_dir = os.path.join(local_path, module_name)
    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    updates = {name: tensor.detach().to(device="cpu") for name, tensor in named_tensors}
    found: set[str] = set()

    def _iter_expected():
        for name, tensor in safetensors_weights_iterator(safetensors_files):
            if name in updates:
                found.add(name)
                yield name, updates[name]
            else:
                yield name, tensor

    checksum = compute_weights_checksum(_iter_expected())
    missing = sorted(set(updates.keys()) - found)
    assert not missing, f"Updated parameter(s) not found in module '{module_name}': {missing}"
    return checksum


def _iter_module_named_tensors(
    model_path: str,
    module_name: str,
):
    """Iterate all (name, tensor) pairs under a specific module directory."""
    local_path = maybe_download_model(model_path)
    weights_dir = os.path.join(local_path, module_name)
    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"
    yield from safetensors_weights_iterator(safetensors_files)


def _append_perf_log(line: str) -> None:
    os.makedirs(os.path.dirname(_PERF_LOG_PATH), exist_ok=True)
    with open(_PERF_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _append_multi_gpu_profiling_log(line: str) -> None:
    os.makedirs(os.path.dirname(_MULTI_GPU_PROFILING_LOG_PATH), exist_ok=True)
    with open(_MULTI_GPU_PROFILING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _append_tp_debug_log(line: str) -> None:
    os.makedirs(os.path.dirname(_TP_DEBUG_LOG_PATH), exist_ok=True)
    with open(_TP_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _kill_existing_sglang_serve_processes() -> None:
    """Best-effort cleanup to avoid stale local servers affecting tests."""
    subprocess.run(
        ['bash', '-lc', 'pkill -f "sglang serve" || true'],
        check=False,
    )


def _build_direct_update_ranked(
    model_path: str,
    module_name: str,
    rank: int,
    value_base: float = 2.0,
    value_sequence: list[float] | None = None,
    num_params: int = 5,
    max_numel: int | None = None,
    min_numel: int | None = None,
    param_names: list[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Build a full-tensor payload on one GPU for one TP rank.

    The payload tensors are full (unsharded) tensors. Sharding happens in server-side
    loading logic for TP. Values are rank-coded to distinguish TP-rank inputs.
    """
    rank_value = value_base + float(rank)
    named_tensors = _build_direct_update(
        model_path=model_path,
        module_name=module_name,
        value=rank_value,
        value_sequence=value_sequence,
        num_params=num_params,
        max_numel=max_numel,
        min_numel=min_numel,
        param_names=param_names,
    )
    return named_tensors


def _make_param_like_for_loader(
    actual_param: torch.nn.Parameter, tensor: torch.Tensor
) -> torch.nn.Parameter:
    cls = actual_param.__class__
    try:
        new_param = cls.__new__(cls, tensor, requires_grad=False)
    except TypeError:
        new_param = cls.__new__(cls, tensor)
    new_param.__dict__.update(actual_param.__dict__)
    new_param.requires_grad = False
    return new_param


def _build_reference_param_dict_for_transformer(
    model_path: str,
    tp_size: int,
    target_param_names: set[str] | None = None,
) -> dict[str, torch.nn.Parameter]:
    """Build a reference transformer model (meta) to access param weight_loader."""
    server_args = ServerArgs.from_kwargs(
        model_path=model_path,
        num_gpus=tp_size,
        tp_size=tp_size,
    )
    # Attention backend selection during model init reads global server args.
    set_global_server_args(server_args)
    local_model_path = maybe_download_model(model_path)
    component_model_path = os.path.join(local_model_path, _TRANSFORMER_MODULE)
    hf_config = get_diffusers_component_config(component_path=component_model_path)
    dit_config = server_args.pipeline_config.dit_config
    dit_config.update_model_arch(hf_config)
    cls_name = hf_config.pop("_class_name")
    model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
    init_params = {"config": dit_config, "hf_config": hf_config, "quant_config": None}
    with set_default_torch_dtype(torch.bfloat16), torch.device("meta"):
        model = model_cls(**init_params)
    named_params = dict(model.named_parameters())
    if target_param_names is None:
        return named_params
    return {k: v for k, v in named_params.items() if k in target_param_names}


def _compute_expected_rank_shard_sha(
    named_tensors: list[tuple[str, torch.Tensor]],
    param_dict: dict[str, torch.nn.Parameter],
) -> tuple[
    dict[str, str],
    dict[str, int],
    dict[str, tuple[int, ...]],
    dict[str, list[float]],
    dict[str, tuple[float, float, float]],
]:
    """Compute expected rank-local shard SHA256 via runtime load path.

    This mirrors the core update chain:
    _apply_weights -> _load_weights_into_module -> load_weights_into_model
    """
    # Build a materialized shadow param dict carrying original parameter attrs
    # (including weight_loader metadata), then run load_weights_into_model on it.
    # Only keep the updated parameter subset to mirror the runtime's per-name path.
    shadow_param_dict: dict[str, torch.nn.Parameter] = {}
    for name, _ in named_tensors:
        if name not in param_dict:
            continue
        meta_param = param_dict[name]
        shadow_tensor = torch.empty(
            meta_param.shape,
            device="cuda",
            dtype=meta_param.dtype,
        )
        shadow_param_dict[name] = _make_param_like_for_loader(meta_param, shadow_tensor)

    load_weights_into_model(
        weights_iter=iter(named_tensors),
        model_params=shadow_param_dict,
        module_name=_TRANSFORMER_MODULE,
    )

    expected_sha256: dict[str, str] = {}
    expected_numel: dict[str, int] = {}
    expected_shape: dict[str, tuple[int, ...]] = {}
    expected_head_values: dict[str, list[float]] = {}
    expected_stats: dict[str, tuple[float, float, float]] = {}
    for name, _ in named_tensors:
        if name not in shadow_param_dict:
            continue
        expected_shape[name] = tuple(shadow_param_dict[name].shape)
        expected_flat = (
            shadow_param_dict[name]
            .detach()
            .reshape(-1)
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )

        expected_sha256[name] = hashlib.sha256(
            expected_flat.numpy().tobytes()
        ).hexdigest()
        expected_numel[name] = int(expected_flat.numel())
        head = expected_flat[: min(8, expected_flat.numel())]
        expected_head_values[name] = head.tolist()
        expected_stats[name] = (
            float(expected_flat.min().item()),
            float(expected_flat.max().item()),
            float(expected_flat.mean().item()),
        )
    return (
        expected_sha256,
        expected_numel,
        expected_shape,
        expected_head_values,
        expected_stats,
    )


def _tp_payload_worker(
    rank: int,
    world_size: int,
    dist_init_method: str,
    default_model: str,
    module_name: str,
    num_params: int,
    max_numel: int | None,
    min_numel: int | None,
    value_base: float,
    value_sequence: list[float] | None,
    param_names: list[str] | None,
    queue,
    done_event,
) -> None:
    """Worker for building rank-local CUDA payload and gathering to rank 0 via gloo."""
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"Need at least {world_size} CUDA devices to build TP payload "
                f"(available={torch.cuda.device_count()})"
            )
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="gloo",
            init_method=dist_init_method,
            rank=rank,
            world_size=world_size,
        )

        named_tensors = _build_direct_update_ranked(
            model_path=default_model,
            module_name=module_name,
            rank=rank,
            value_base=value_base,
            value_sequence=value_sequence,
            num_params=num_params,
            max_numel=max_numel,
            min_numel=min_numel,
            param_names=param_names,
        )
        named_tensors = _maybe_filter_tp_named_tensors(named_tensors)
        serialized_payload = MultiprocessingSerializer.serialize(
            named_tensors, output_str=True
        )
        gathered_payloads = [None] * world_size if rank == 0 else None
        dist.gather_object(serialized_payload, gathered_payloads, dst=0)

        if rank == 0:
            queue.put(
                {
                    "serialized_payloads": gathered_payloads,
                }
            )

        # Keep producer processes and CUDA tensors alive until parent signals completion.
        done_event.wait(timeout=1200)
    except Exception as e:
        if rank == 0:
            queue.put({"error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@dataclass
class _TPPayloadSession:
    serialized_payloads: list[str]
    procs: list
    done_event: Any

    def close(self) -> None:
        self.done_event.set()
        for p in self.procs:
            p.join(timeout=120)
        for p in self.procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        bad = [p.exitcode for p in self.procs if p.exitcode not in (0, None)]
        assert not bad, f"TP payload builder worker exited with non-zero code(s): {bad}"


def _build_tp_payloads_via_gloo_gather(
    default_model: str,
    module_name: str,
    tp_size: int = 4,
    value_base: float = 2.0,
    value_sequence: list[float] | None = None,
    num_params: int = 3,
    max_numel: int = 65536,
    min_numel: int | None = None,
    param_names: list[str] | None = None,
) -> _TPPayloadSession:
    """Build TP payloads with one process/GPU per TP rank and gather to rank 0.

    The child processes are intentionally kept alive until session.close() is called,
    to keep CUDA IPC-backed tensors valid during server-side deserialization.
    """
    from sglang.multimodal_gen.test.test_utils import find_free_port

    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    done_event = ctx.Event()
    port = find_free_port()
    dist_init_method = f"tcp://127.0.0.1:{port}"

    procs = []
    for rank in range(tp_size):
        p = ctx.Process(
            target=_tp_payload_worker,
            args=(
                rank,
                tp_size,
                dist_init_method,
                default_model,
                module_name,
                num_params,
                max_numel,
                min_numel,
                value_base,
                value_sequence,
                param_names,
                result_queue,
                done_event,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    result = result_queue.get(timeout=600)
    assert "error" not in result, f"Failed to build TP payloads: {result['error']}"

    payloads = result["serialized_payloads"]
    assert len(payloads) == tp_size, (
        f"Expected gathered payload size {tp_size}, got {len(payloads)}"
    )
    return _TPPayloadSession(
        serialized_payloads=payloads,
        procs=procs,
        done_event=done_event,
    )


def _set_dist_env_for_rank(
    rank: int, world_size: int, dist_init_method: str
) -> None:
    # dist_init_method format in this test is tcp://127.0.0.1:<port>
    addr_port = dist_init_method.split("://", 1)[1]
    master_addr, master_port = addr_port.rsplit(":", 1)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port


def _tp_expected_sha_worker(
    rank: int,
    world_size: int,
    dist_init_method: str,
    default_model: str,
    module_name: str,
    num_params: int,
    max_numel: int | None,
    min_numel: int | None,
    value_base: float,
    value_sequence: list[float] | None,
    queue,
    serialized_payload: str | None = None,
) -> None:
    """Worker for building expected rank-local shard SHA via TP/NCCL runtime path."""
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"Need at least {world_size} CUDA devices to build TP expected "
                f"(available={torch.cuda.device_count()})"
            )
        torch.cuda.set_device(rank)
        _set_dist_env_for_rank(
            rank=rank, world_size=world_size, dist_init_method=dist_init_method
        )
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=world_size,
            enable_cfg_parallel=False,
            ulysses_degree=1,
            ring_degree=1,
            sp_size=1,
            dp_size=1,
            distributed_init_method=dist_init_method,
            dist_timeout=300,
        )

        if serialized_payload is not None:
            named_tensors = MultiprocessingSerializer.deserialize(serialized_payload)
        else:
            named_tensors = _build_direct_update_ranked(
                model_path=default_model,
                module_name=module_name,
                rank=rank,
                value_base=value_base,
                value_sequence=value_sequence,
                num_params=num_params,
                max_numel=max_numel,
                min_numel=min_numel,
            )
            named_tensors = _maybe_filter_tp_named_tensors(named_tensors)
        full_shape = {name: tuple(t.shape) for name, t in named_tensors}
        target_param_names = {name for name, _ in named_tensors}
        ref_param_dict = _build_reference_param_dict_for_transformer(
            model_path=default_model,
            tp_size=world_size,
            target_param_names=target_param_names,
        )
        (
            expected_sha256,
            expected_numel,
            expected_shape,
            expected_head_values,
            expected_stats,
        ) = _compute_expected_rank_shard_sha(named_tensors=named_tensors, param_dict=ref_param_dict)
        queue.put(
            {
                "rank": rank,
                "expected_sha256": expected_sha256,
                "expected_numel": expected_numel,
                "expected_shape": expected_shape,
                "expected_head_values": expected_head_values,
                "expected_stats": expected_stats,
                "full_shape": full_shape,
            }
        )
    except Exception as e:
        queue.put({"error": f"{type(e).__name__}: {e}"})
        raise


def _build_tp_expected_sha_via_nccl(
    default_model: str,
    module_name: str,
    serialized_payloads: list[str] | None = None,
    tp_size: int = 4,
    value_base: float = 2.0,
    value_sequence: list[float] | None = None,
    num_params: int = 3,
    max_numel: int = 65536,
    min_numel: int | None = None,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, int]],
    list[dict[str, tuple[int, ...]]],
    list[dict[str, list[float]]],
    list[dict[str, tuple[float, float, float]]],
    list[dict[str, tuple[int, ...]]],
]:
    """Build per-rank expected shard SHA/numel using TP/NCCL runtime load path."""
    from sglang.multimodal_gen.test.test_utils import find_free_port

    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    port = find_free_port()
    dist_init_method = f"tcp://127.0.0.1:{port}"

    procs = []
    for rank in range(tp_size):
        serialized_payload = (
            serialized_payloads[rank] if serialized_payloads is not None else None
        )
        p = ctx.Process(
            target=_tp_expected_sha_worker,
            args=(
                rank,
                tp_size,
                dist_init_method,
                default_model,
                module_name,
                num_params,
                max_numel,
                min_numel,
                value_base,
                value_sequence,
                result_queue,
            ),
            kwargs={"serialized_payload": serialized_payload},
            daemon=False,
        )
        p.start()
        procs.append(p)

    expected_sha256_by_rank: list[dict[str, str] | None] = [None] * tp_size
    expected_numel_by_rank: list[dict[str, int] | None] = [None] * tp_size
    expected_shape_by_rank: list[dict[str, tuple[int, ...]] | None] = [None] * tp_size
    expected_head_values_by_rank: list[dict[str, list[float]] | None] = [None] * tp_size
    expected_stats_by_rank: list[dict[str, tuple[float, float, float]] | None] = [None] * tp_size
    full_shape_by_rank: list[dict[str, tuple[int, ...]] | None] = [None] * tp_size
    for _ in range(tp_size):
        result = result_queue.get(timeout=600)
        assert "error" not in result, (
            f"Failed to build TP expected sha: {result['error']}"
        )
        rank = int(result["rank"])
        expected_sha256_by_rank[rank] = result["expected_sha256"]
        expected_numel_by_rank[rank] = result["expected_numel"]
        expected_shape_by_rank[rank] = result["expected_shape"]
        expected_head_values_by_rank[rank] = result["expected_head_values"]
        expected_stats_by_rank[rank] = result["expected_stats"]
        full_shape_by_rank[rank] = result["full_shape"]

    for p in procs:
        p.join(timeout=180)
    bad = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    assert not bad, f"TP expected-sha worker exited with non-zero code(s): {bad}"
    assert all(x is not None for x in expected_sha256_by_rank)
    assert all(x is not None for x in expected_numel_by_rank)
    assert all(x is not None for x in expected_shape_by_rank)
    assert all(x is not None for x in expected_head_values_by_rank)
    assert all(x is not None for x in expected_stats_by_rank)
    assert all(x is not None for x in full_shape_by_rank)
    return (
        expected_sha256_by_rank,  # type: ignore[return-value]
        expected_numel_by_rank,  # type: ignore[return-value]
        expected_shape_by_rank,  # type: ignore[return-value]
        expected_head_values_by_rank,  # type: ignore[return-value]
        expected_stats_by_rank,  # type: ignore[return-value]
        full_shape_by_rank,  # type: ignore[return-value]
    )


def _tp_expected_module_checksum_worker(
    rank: int,
    world_size: int,
    dist_init_method: str,
    default_model: str,
    module_name: str,
    queue,
    serialized_payload: str,
) -> None:
    """Worker for expected rank-local module checksum via TP load path."""
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"Need at least {world_size} CUDA devices to build TP expected "
                f"(available={torch.cuda.device_count()})"
            )
        torch.cuda.set_device(rank)
        _set_dist_env_for_rank(
            rank=rank, world_size=world_size, dist_init_method=dist_init_method
        )
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=world_size,
            enable_cfg_parallel=False,
            ulysses_degree=1,
            ring_degree=1,
            sp_size=1,
            dp_size=1,
            distributed_init_method=dist_init_method,
            dist_timeout=300,
        )

        ref_param_dict = _build_reference_param_dict_for_transformer(
            model_path=default_model,
            tp_size=world_size,
            target_param_names=None,
        )
        shadow_param_dict: dict[str, torch.nn.Parameter] = {}
        for name, meta_param in ref_param_dict.items():
            shadow_tensor = torch.empty(
                meta_param.shape,
                device="cuda",
                dtype=meta_param.dtype,
            )
            shadow_param_dict[name] = _make_param_like_for_loader(meta_param, shadow_tensor)

        # Baseline: full module from disk.
        load_weights_into_model(
            weights_iter=_iter_module_named_tensors(default_model, module_name),
            model_params=shadow_param_dict,
            module_name=module_name,
        )
        # Apply this rank's update payload.
        named_tensors = MultiprocessingSerializer.deserialize(serialized_payload)
        named_tensors = _maybe_filter_tp_named_tensors(named_tensors)
        load_weights_into_model(
            weights_iter=iter(named_tensors),
            model_params=shadow_param_dict,
            module_name=module_name,
        )

        expected_module_checksum = compute_weights_checksum(shadow_param_dict.items())
        queue.put({"rank": rank, "expected_module_checksum": expected_module_checksum})
    except Exception as e:
        queue.put({"error": f"{type(e).__name__}: {e}"})
        raise


def _build_tp_expected_module_checksum_via_nccl(
    default_model: str,
    module_name: str,
    serialized_payloads: list[str],
    tp_size: int,
) -> list[str]:
    """Build per-rank expected module checksums using TP/NCCL runtime load path."""
    from sglang.multimodal_gen.test.test_utils import find_free_port

    assert len(serialized_payloads) == tp_size, (
        f"Expected gathered payload size {tp_size}, got {len(serialized_payloads)}"
    )

    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    port = find_free_port()
    dist_init_method = f"tcp://127.0.0.1:{port}"

    procs = []
    for rank in range(tp_size):
        p = ctx.Process(
            target=_tp_expected_module_checksum_worker,
            args=(
                rank,
                tp_size,
                dist_init_method,
                default_model,
                module_name,
                result_queue,
                serialized_payloads[rank],
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    expected_module_checksum_by_rank: list[str | None] = [None] * tp_size
    for _ in range(tp_size):
        result = result_queue.get(timeout=1200)
        assert "error" not in result, (
            f"Failed to build TP expected module checksum: {result['error']}"
        )
        rank = int(result["rank"])
        expected_module_checksum_by_rank[rank] = result["expected_module_checksum"]

    for p in procs:
        p.join(timeout=1800)
    bad = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    assert not bad, f"TP expected-checksum worker exited with non-zero code(s): {bad}"
    assert all(x is not None for x in expected_module_checksum_by_rank)
    return expected_module_checksum_by_rank  # type: ignore[return-value]


class _UpdateWeightsFromTensorApiMixin:
    def _update_weights_from_disk(
        self,
        base_url: str,
        model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
        timeout: int = 300,
    ) -> tuple[dict, int]:
        payload = {"model_path": model_path, "flush_cache": flush_cache}
        if target_modules is not None:
            payload["target_modules"] = target_modules
        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json=payload,
            timeout=timeout,
        )
        return response.json(), response.status_code

    def _update_weights_from_tensor(
        self,
        base_url: str,
        named_tensors,
        load_format: str | None = None,
        target_modules: list[str] | None = None,
        weight_version: str | None = None,
        serialized_payloads: list[str] | None = None,
        timeout: int = 300,
    ) -> tuple[dict, int]:
        payload = {
            "serialized_named_tensors": serialized_payloads
            if serialized_payloads is not None
            else [MultiprocessingSerializer.serialize(named_tensors, output_str=True)]
        }
        if load_format is not None:
            payload["load_format"] = load_format
        if target_modules is not None:
            payload["target_modules"] = target_modules
        if weight_version is not None:
            payload["weight_version"] = weight_version

        t_start = time.perf_counter()
        response = requests.post(
            f"{base_url}/update_weights_from_tensor",
            json=payload,
            timeout=timeout,
        )
        t_end = time.perf_counter()
        response_json = response.json()
        profiling = response_json.get("profiling")
        _append_perf_log(
            (
                f"update_weights_from_tensor "
                f"base_url={base_url} "
                f"load_format={load_format} "
                f"target_modules={target_modules} "
                f"status={response.status_code} "
                f"elapsed_s={t_end - t_start:.6f}"
            )
        )
        if isinstance(profiling, dict):
            _append_perf_log(
                (
                    "update_weights_from_tensor_profiling "
                    f"request_id={profiling.get('request_id')} "
                    f"load_format={profiling.get('load_format')} "
                    f"deserialize_s={profiling.get('deserialize_s')} "
                    f"deflatten_meta_convert_s={profiling.get('deflatten_meta_convert_s')} "
                    f"deflatten_reconstruct_s={profiling.get('deflatten_reconstruct_s')} "
                    f"deflatten_total_s={profiling.get('deflatten_total_s')} "
                    f"load_s={profiling.get('load_s')} "
                    f"total_s={profiling.get('total_s')}"
                )
            )
        return response_json, response.status_code

    def _get_weights_checksum(
        self,
        base_url: str,
        module_names: list[str] | None = None,
        timeout: int = 300,
    ) -> dict:
        payload = {}
        if module_names is not None:
            payload["module_names"] = module_names
        response = requests.post(
            f"{base_url}/get_weights_checksum",
            json=payload,
            timeout=timeout,
        )
        assert (
            response.status_code == 200
        ), f"get_weights_checksum failed: {response.status_code} {response.text}"
        return response.json()

    def _get_server_info(self, base_url: str, timeout: int = 30) -> dict:
        response = requests.get(
            f"{base_url}/server_info",
            timeout=timeout,
        )
        assert (
            response.status_code == 200
        ), f"get_server_info failed: {response.status_code} {response.text}"
        return response.json()

    def _get_process_tree_pids(self, root_pid: int) -> list[int]:
        """Return root + descendant PIDs from the OS process tree."""
        try:
            result = subprocess.run(
                ["ps", "-e", "-o", "pid=,ppid="],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return [root_pid]

        children_by_parent: dict[int, list[int]] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except ValueError:
                continue
            children_by_parent.setdefault(ppid, []).append(pid)

        stack = [root_pid]
        visited: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(children_by_parent.get(cur, []))
        return sorted(visited)

    def _get_process_tree_gpu_memory_bytes(
        self, root_pid: int
    ) -> tuple[int | None, list[int], list[int]]:
        """Return (bytes, tree_pids, matched_gpu_pids) for root + descendants.

        bytes is None only when GPU memory cannot be queried (e.g. nvidia-smi missing).
        """
        tree_pids = self._get_process_tree_pids(root_pid)

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None, tree_pids, []

        tree_pid_set = set(tree_pids)
        matched_gpu_pids: set[int] = set()
        total_mib = 0
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                row_pid = int(parts[0])
                used_mib = int(parts[1])
            except ValueError:
                continue
            if row_pid in tree_pid_set:
                total_mib += used_mib
                matched_gpu_pids.add(row_pid)
        return total_mib * 1024 * 1024, tree_pids, sorted(matched_gpu_pids)

    def _reset_to_base_model(self, base_url: str, default_model: str) -> None:
        result, status_code = self._update_weights_from_disk(
            base_url,
            default_model,
            flush_cache=True,
        )
        assert status_code == 200, f"Failed to reset to base model: {result}"
        assert result.get("success", False), f"Failed to reset to base model: {result}"

class TestUpdateWeightsFromTensor(_UpdateWeightsFromTensorApiMixin):
    @pytest.fixture(
        scope="class",
        params=_ACTIVE_MODEL_PAIRS,
        ids=_PAIR_IDS,
    )
    def diffusion_server_no_offload(self, request):
        default_model, _ = request.param
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))
        extra_args = "--num-gpus 1"
        dit_cpu_offload_env = os.environ.get(_DIT_CPU_OFFLOAD_ENV)
        if dit_cpu_offload_env is not None and dit_cpu_offload_env.lower() in (
            "0",
            "false",
            "no",
        ):
            extra_args += " --dit-cpu-offload false"

        maybe_download_model(default_model)
        _kill_existing_sglang_serve_processes()

        manager = ServerManager(
            model=default_model,
            port=port,
            wait_deadline=wait_deadline,
            extra_args=extra_args,
        )
        ctx = manager.start()

        try:
            yield ctx, default_model
        finally:
            ctx.cleanup()

    def test_update_weights_from_tensor_direct(
        self, diffusion_server_no_offload
    ):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)
        before = self._get_weights_checksum(base_url)

        payload = _build_direct_update(default_model, _TRANSFORMER_MODULE, value=2.0)
        expected_transformer_checksum = _compute_expected_checksum_after_direct_update(
            model_path=default_model,
            module_name=_TRANSFORMER_MODULE,
            named_tensors=payload,
        )
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=payload,
            target_modules=[_TRANSFORMER_MODULE],
        )

        assert status_code == 200, f"Expected 200, got {status_code}: {result}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        after = self._get_weights_checksum(base_url)
        assert set(after.keys()) == set(before.keys()), (
            "Module set changed unexpectedly after update.\n"
            f"before_only={sorted(set(before.keys()) - set(after.keys()))}\n"
            f"after_only={sorted(set(after.keys()) - set(before.keys()))}"
        )
        assert after.get(_TRANSFORMER_MODULE) == expected_transformer_checksum, (
            f"Expected transformer checksum to match direct-update payload\n"
            f"  expected: {expected_transformer_checksum}\n"
            f"  actual:   {after.get(_TRANSFORMER_MODULE)}"
        )
        for name in sorted(after.keys()):
            if name == _TRANSFORMER_MODULE:
                continue
            assert after.get(name) == before.get(name), (
                f"Non-targeted module '{name}' should be unchanged\n"
                f"  before: {before.get(name)}\n"
                f"  after:  {after.get(name)}"
            )

    def test_update_weights_from_tensor_flattened_bucket(self, diffusion_server_no_offload):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)
        before = self._get_weights_checksum(base_url)

        direct_payload = _build_direct_update(default_model, _TRANSFORMER_MODULE, value=2.0)
        expected_transformer_checksum = _compute_expected_checksum_after_direct_update(
            model_path=default_model,
            module_name=_TRANSFORMER_MODULE,
            named_tensors=direct_payload,
        )
        bucket = FlattenedTensorBucket(named_tensors=direct_payload)
        bucket_payload = {
            _TRANSFORMER_MODULE: {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            }
        }

        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=bucket_payload,
            load_format="flattened_bucket",
            target_modules=[_TRANSFORMER_MODULE],
        )

        assert status_code == 200, f"Expected 200, got {status_code}: {result}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        after = self._get_weights_checksum(base_url)
        assert set(after.keys()) == set(before.keys()), (
            "Module set changed unexpectedly after update.\n"
            f"before_only={sorted(set(before.keys()) - set(after.keys()))}\n"
            f"after_only={sorted(set(after.keys()) - set(before.keys()))}"
        )
        assert after.get(_TRANSFORMER_MODULE) == expected_transformer_checksum, (
            "Expected transformer checksum to match flattened-bucket payload\n"
            f"  expected: {expected_transformer_checksum}\n"
            f"  actual:   {after.get(_TRANSFORMER_MODULE)}"
        )
        for name in sorted(after.keys()):
            if name == _TRANSFORMER_MODULE:
                continue
            assert after.get(name) == before.get(name), (
                f"Non-targeted module '{name}' should be unchanged\n"
                f"  before: {before.get(name)}\n"
                f"  after:  {after.get(name)}"
            )

    def test_update_weights_specific_modules(self, diffusion_server_no_offload):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)
        before = self._get_weights_checksum(base_url)

        target_module = _TRANSFORMER_MODULE
        assert before.get(target_module) != "not_found", (
            f"Target module '{target_module}' not found. checksums={before}"
        )
        target_payload = _build_direct_update(default_model, target_module, value=2.0)
        expected_target_checksum = _compute_expected_checksum_after_direct_update(
            model_path=default_model,
            module_name=target_module,
            named_tensors=target_payload,
        )
        payload = {target_module: target_payload}
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=payload,
            target_modules=[target_module],
        )

        assert status_code == 200, f"Update failed: {result}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        after = self._get_weights_checksum(base_url)
        assert set(after.keys()) == set(before.keys()), (
            "Module set changed unexpectedly after update.\n"
            f"before_only={sorted(set(before.keys()) - set(after.keys()))}\n"
            f"after_only={sorted(set(after.keys()) - set(before.keys()))}"
        )
        assert after.get(target_module) == expected_target_checksum, (
            f"Expected target module checksum to match payload for '{target_module}'\n"
            f"  expected: {expected_target_checksum}\n"
            f"  actual:   {after.get(target_module)}"
        )

        for name in sorted(after.keys()):
            if name == target_module:
                continue
            assert after.get(name) == before.get(name), (
                f"Non-targeted module '{name}' should be unchanged\n"
                f"  before: {before.get(name)}\n"
                f"  after:  {after.get(name)}"
            )

    def test_update_weights_missing_serialized_named_tensors(
        self, diffusion_server_no_offload
    ):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)
        before = self._get_weights_checksum(base_url, module_names=[_TRANSFORMER_MODULE])

        response = requests.post(
            f"{base_url}/update_weights_from_tensor",
            json={},
            timeout=30,
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        result = response.json()
        assert not result.get("success", True)
        assert "serialized_named_tensors is required" in result.get("message", "")

        after = self._get_weights_checksum(base_url, module_names=[_TRANSFORMER_MODULE])
        assert after.get(_TRANSFORMER_MODULE) == before.get(_TRANSFORMER_MODULE)

    def test_update_weights_invalid_tp_payload_size(self, diffusion_server_no_offload):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        server_info = self._get_server_info(base_url)
        tp_size = int(server_info.get("tp_size", 1))
        if tp_size == 2:
            pytest.skip(
                "test_update_weights_invalid_tp_payload_size is invalid when tp_size=2; "
                "len(serialized_named_tensors)=2 is a valid payload size. "
                "Run this test with tp_size != 2."
            )

        self._reset_to_base_model(base_url, default_model)
        before = self._get_weights_checksum(base_url, module_names=[_TRANSFORMER_MODULE])

        one_payload = MultiprocessingSerializer.serialize(
            _build_direct_update(default_model, _TRANSFORMER_MODULE, value=2.0),
            output_str=True,
        )
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=None,
            serialized_payloads=[one_payload, one_payload],
            target_modules=[_TRANSFORMER_MODULE],
        )

        assert status_code == 400, f"Expected 400, got {status_code}: {result}"
        assert not result.get("success", True)
        assert "serialized_named_tensors size must be 1 or tp_size" in result.get(
            "message", ""
        )

        after = self._get_weights_checksum(base_url, module_names=[_TRANSFORMER_MODULE])
        assert after.get(_TRANSFORMER_MODULE) == before.get(_TRANSFORMER_MODULE)

    def test_update_weights_default_target_transformer(self, diffusion_server_no_offload):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)
        before = self._get_weights_checksum(base_url)

        payload = _build_direct_update(default_model, _TRANSFORMER_MODULE, value=2.0)
        expected_transformer_checksum = _compute_expected_checksum_after_direct_update(
            model_path=default_model,
            module_name=_TRANSFORMER_MODULE,
            named_tensors=payload,
        )
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=payload,
            target_modules=None,
        )

        assert status_code == 200, f"Expected 200, got {status_code}: {result}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        after = self._get_weights_checksum(base_url)
        assert set(after.keys()) == set(before.keys()), (
            "Module set changed unexpectedly after update.\n"
            f"before_only={sorted(set(before.keys()) - set(after.keys()))}\n"
            f"after_only={sorted(set(after.keys()) - set(before.keys()))}"
        )
        assert after.get(_TRANSFORMER_MODULE) == expected_transformer_checksum, (
            "Expected transformer checksum to match payload when target_modules is omitted\n"
            f"  expected: {expected_transformer_checksum}\n"
            f"  actual:   {after.get(_TRANSFORMER_MODULE)}"
        )
        for name in sorted(after.keys()):
            if name == _TRANSFORMER_MODULE:
                continue
            assert after.get(name) == before.get(name), (
                f"Non-transformer module '{name}' should be unchanged\n"
                f"  before: {before.get(name)}\n"
                f"  after:  {after.get(name)}"
            )

    def test_update_weights_nonexistent_module(self, diffusion_server_no_offload):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)

        payload = _build_direct_update(default_model, _TRANSFORMER_MODULE, value=2.0)
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=payload,
            target_modules=["nonexistent_module"],
            timeout=60,
        )
        logger.info("Update nonexistent module result: %s", result)

        assert status_code == 400, f"Expected 400, got {status_code}"
        assert not result.get("success", True)
        assert "not found in pipeline" in result.get("message", "")

    def test_corrupted_tensor_update_fails(self, diffusion_server_no_offload):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)
        # Corrupted payload: transformer has invalid shape.
        corrupted_payload = _build_invalid_shape_update(
            default_model, _TRANSFORMER_MODULE
        )
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=corrupted_payload,
            target_modules=[_TRANSFORMER_MODULE],
        )
        assert status_code == 400, f"Expected 400 on corrupted update, got {status_code}"
        assert not result.get("success", True)
        message = result.get("message", "")
        assert "Failed to update module 'transformer'" in message
        assert "partially updated" in message.lower()

    def test_update_weights_from_tensor_memory_leak(
        self, diffusion_server_no_offload
    ):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"
        server_pid = ctx.process.pid

        self._reset_to_base_model(base_url, default_model)
        payload = _build_direct_update(
            default_model,
            _TRANSFORMER_MODULE,
            value=2.0,
            num_params=5,
            max_numel=65536,
        )

        # Warm up once to reduce allocator startup noise.
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=payload,
            target_modules=[_TRANSFORMER_MODULE],
            timeout=120,
        )
        assert status_code == 200, f"Warmup update failed: {status_code}, {result}"
        assert result.get("success", False), f"Warmup update failed: {result}"
        time.sleep(1.0)

        memory_before, tree_pids_before, matched_gpu_pids_before = (
            self._get_process_tree_gpu_memory_bytes(server_pid)
        )
        if memory_before is None:
            pytest.skip(
                "Cannot measure server GPU memory via nvidia-smi in this env "
                f"(server_pid={server_pid})."
            )

        num_updates = 5
        for _ in range(num_updates):
            result, status_code = self._update_weights_from_tensor(
                base_url,
                named_tensors=payload,
                target_modules=[_TRANSFORMER_MODULE],
                timeout=120,
            )
            assert status_code == 200, f"Update failed: {status_code}, {result}"
            assert result.get("success", False), f"Update failed: {result}"

        time.sleep(1.0)
        memory_after, tree_pids_after, matched_gpu_pids_after = (
            self._get_process_tree_gpu_memory_bytes(server_pid)
        )
        if memory_after is None:
            pytest.skip(
                "Cannot measure server GPU memory via nvidia-smi in this env "
                f"(server_pid={server_pid})."
            )

        # Allow small allocator/caching fluctuation while catching persistent growth.
        tolerance_bytes = 256 * 1024 * 1024
        assert memory_after <= memory_before + tolerance_bytes, (
            f"Potential GPU memory regression after repeated tensor updates. "
            f"before={memory_before}B after={memory_after}B "
            f"delta={memory_after - memory_before}B "
            f"tolerance={tolerance_bytes}B "
            f"server_pid={server_pid} "
            f"tree_before={tree_pids_before} tree_after={tree_pids_after} "
            f"gpu_pids_before={matched_gpu_pids_before} "
            f"gpu_pids_after={matched_gpu_pids_after}"
        )

    def test_update_weights_from_tensor_flattened_bucket_full_transformer(
        self, diffusion_server_no_offload
    ):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        # Ensure the server starts from the base model.
        self._reset_to_base_model(base_url, default_model)
        base_transformer_checksum = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        ).get(_TRANSFORMER_MODULE)
        assert base_transformer_checksum, "Missing base transformer checksum"

        # Perturb weights first so this test validates a real restore path.
        perturb_payload = _build_direct_update(
            default_model, _TRANSFORMER_MODULE, value=11.0, num_params=3
        )
        perturb_result, perturb_status = self._update_weights_from_tensor(
            base_url,
            named_tensors=perturb_payload,
            target_modules=[_TRANSFORMER_MODULE],
            timeout=600,
        )
        assert perturb_status == 200, (
            f"Failed to perturb transformer before restore test: {perturb_result}"
        )
        assert perturb_result.get("success", False), (
            f"Failed to perturb transformer before restore test: {perturb_result}"
        )
        perturbed_checksum = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        ).get(_TRANSFORMER_MODULE)
        assert perturbed_checksum != base_transformer_checksum, (
            "Precondition failed: perturb update did not change transformer checksum"
        )

        bucket_size_bytes = int(
            os.environ.get(_BUCKET_SIZE_ENV, str(4096 * 1024 * 1024))
        )
        perf_tag = "single_gpu_flattened_bucket_full_transformer"
        test_name = (
            "TestUpdateWeightsFromTensor."
            "test_update_weights_from_tensor_flattened_bucket_full_transformer"
        )

        current_bucket: list[tuple[str, torch.Tensor]] = []
        current_bucket_bytes = 0
        total_tensors = 0
        total_payload_bytes = 0
        total_request_elapsed_s = 0.0
        bucket_idx = 0

        def _flush_bucket():
            nonlocal current_bucket, current_bucket_bytes
            nonlocal bucket_idx, total_payload_bytes, total_request_elapsed_s
            if not current_bucket:
                return

            bucket = FlattenedTensorBucket(named_tensors=current_bucket)
            bucket_payload = {
                _TRANSFORMER_MODULE: {
                    "flattened_tensor": bucket.get_flattened_tensor(),
                    "metadata": bucket.get_metadata(),
                }
            }
            request_t0 = time.perf_counter()
            result, status_code = self._update_weights_from_tensor(
                base_url,
                named_tensors=bucket_payload,
                load_format="flattened_bucket",
                target_modules=[_TRANSFORMER_MODULE],
                weight_version=f"{perf_tag}-bucket-{bucket_idx}",
                timeout=1800,
            )
            request_elapsed_s = time.perf_counter() - request_t0
            assert status_code == 200, (
                f"Bucket {bucket_idx} failed, expected 200 got {status_code}: {result}"
            )
            assert result.get("success", False), (
                f"Bucket {bucket_idx} update failed: {result.get('message')}"
            )
            _append_perf_log(
                (
                    f"[{perf_tag}] test={test_name} model={default_model} "
                    f"bucket_idx={bucket_idx} "
                    f"bucket_size_bytes={bucket_size_bytes} "
                    f"bucket_size_gib={bucket_size_bytes / (1024**3):.6f} "
                    f"bucket_bytes={current_bucket_bytes} "
                    f"bucket_gib={current_bucket_bytes / (1024**3):.6f} "
                    f"num_tensors={len(current_bucket)} "
                    f"request_elapsed_s={request_elapsed_s:.6f}"
                )
            )
            total_payload_bytes += current_bucket_bytes
            total_request_elapsed_s += request_elapsed_s
            bucket_idx += 1
            current_bucket = []
            current_bucket_bytes = 0
            torch.cuda.empty_cache()

        _append_perf_log(
            (
                f"[{perf_tag}] start test={test_name} model={default_model} "
                f"bucket_size_bytes={bucket_size_bytes} "
                f"bucket_size_gib={bucket_size_bytes / (1024**3):.6f}"
            )
        )
        for name, tensor in _iter_module_named_tensors(default_model, _TRANSFORMER_MODULE):
            tensor_cuda = tensor.to(device="cuda")
            tensor_bytes = tensor_cuda.numel() * tensor_cuda.element_size()

            if current_bucket and current_bucket_bytes + tensor_bytes > bucket_size_bytes:
                _flush_bucket()

            current_bucket.append((name, tensor_cuda))
            current_bucket_bytes += tensor_bytes
            total_tensors += 1

        _flush_bucket()
        _append_perf_log(
            (
                f"[{perf_tag}] done test={test_name} model={default_model} "
                f"bucket_size_bytes={bucket_size_bytes} "
                f"bucket_size_gib={bucket_size_bytes / (1024**3):.6f} "
                f"num_buckets={bucket_idx} "
                f"total_tensors={total_tensors} "
                f"total_payload_bytes={total_payload_bytes} "
                f"total_payload_gib={total_payload_bytes / (1024**3):.6f} "
                f"total_request_elapsed_s={total_request_elapsed_s:.6f}"
            )
        )

        restored_checksum = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        ).get(_TRANSFORMER_MODULE)
        assert restored_checksum == base_transformer_checksum, (
            "Full-transformer flattened-bucket restore checksum mismatch\n"
            f"  expected(base): {base_transformer_checksum}\n"
            f"  actual:         {restored_checksum}"
        )

# class TestUpdateWeightsFromTensorWithOffload(_UpdateWeightsFromTensorApiMixin):
#     """Test update_weights_from_tensor with layerwise offload enabled."""

#     @pytest.fixture(scope="class", params=_ACTIVE_MODEL_PAIRS, ids=_PAIR_IDS)
#     def diffusion_server_no_offload(self, request):
#         default_model, _ = request.param
#         port = get_dynamic_server_port()
#         wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

#         maybe_download_model(default_model)
#         _kill_existing_sglang_serve_processes()

#         manager = ServerManager(
#             model=default_model,
#             port=port,
#             wait_deadline=wait_deadline,
#             extra_args="--num-gpus 1",
#         )

#         ctx = manager.start()
#         try:
#             yield ctx, default_model
#         finally:
#             ctx.cleanup()

#     @pytest.fixture(scope="class", params=_ACTIVE_MODEL_PAIRS, ids=_PAIR_IDS)
#     def diffusion_server_with_offload(self, request):
#         default_model, _ = request.param
#         port = get_dynamic_server_port()
#         wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

#         maybe_download_model(default_model)
#         _kill_existing_sglang_serve_processes()

#         manager = ServerManager(
#             model=default_model,
#             port=port,
#             wait_deadline=wait_deadline,
#             extra_args="--num-gpus 1 --dit-layerwise-offload true",
#         )

#         ctx = manager.start()
#         try:
#             yield ctx, default_model
#         finally:
#             ctx.cleanup()

#     def test_update_weights_with_offload_enabled(self, diffusion_server_with_offload):
#         ctx, default_model = diffusion_server_with_offload
#         base_url = f"http://localhost:{ctx.port}"

#         self._reset_to_base_model(base_url, default_model)
#         before = self._get_weights_checksum(base_url, module_names=[_TRANSFORMER_MODULE])

#         payload = _build_direct_update(default_model, _TRANSFORMER_MODULE, value=2.0)
#         result, status_code = self._update_weights_from_tensor(
#             base_url,
#             named_tensors=payload,
#             target_modules=[_TRANSFORMER_MODULE],
#         )

#         assert status_code == 200, f"Expected 200, got {status_code}"
#         assert result.get("success", False), f"Update failed: {result.get('message')}"
#         assert "Shape mismatch" not in result.get("message", "")

#         after = self._get_weights_checksum(base_url, module_names=[_TRANSFORMER_MODULE])
#         assert after.get(_TRANSFORMER_MODULE) != before.get(_TRANSFORMER_MODULE)

#     def test_update_weights_from_tensor_direct_full_transformer(
#         self, diffusion_server_no_offload
#     ):
#         ctx, default_model = diffusion_server_no_offload
#         base_url = f"http://localhost:{ctx.port}"

#         # Ensure the server starts from the base model.
#         self._reset_to_base_model(base_url, default_model)

#         perf_tag = "single_gpu_direct_full_transformer"
#         test_name = (
#             "TestUpdateWeightsFromTensorWithOffload."
#             "test_update_weights_from_tensor_direct_full_transformer"
#         )
#         _append_perf_log(f"[{perf_tag}] start test={test_name} model={default_model}")

#         sample_expected: dict[str, list[float]] = {}
#         total_tensors = 0
#         total_payload_bytes = 0
#         total_request_elapsed_s = 0.0
#         request_idx = 0

#         for name, tensor in _iter_module_named_tensors(default_model, _TRANSFORMER_MODULE):
#             tensor_cuda = tensor.to(device="cuda")
#             tensor_bytes = tensor_cuda.numel() * tensor_cuda.element_size()
#             payload = [(name, tensor_cuda)]

#             request_t0 = time.perf_counter()
#             result, status_code = self._update_weights_from_tensor(
#                 base_url,
#                 named_tensors=payload,
#                 target_modules=[_TRANSFORMER_MODULE],
#                 timeout=1800,
#             )
#             request_elapsed_s = time.perf_counter() - request_t0

#             assert status_code == 200, (
#                 f"Request {request_idx} failed, expected 200 got {status_code}: {result}"
#             )
#             assert result.get("success", False), (
#                 f"Request {request_idx} update failed: {result.get('message')}"
#             )

#             _append_perf_log(
#                 (
#                     f"[{perf_tag}] test={test_name} model={default_model} "
#                     f"request_idx={request_idx} "
#                     f"tensor_name={name} "
#                     f"tensor_bytes={tensor_bytes} "
#                     f"request_elapsed_s={request_elapsed_s:.6f}"
#                 )
#             )

#             if len(sample_expected) < 8:
#                 sample_expected[name] = (
#                     tensor_cuda.detach()
#                     .reshape(-1)[:5]
#                     .to(device="cpu", dtype=torch.float32)
#                     .tolist()
#                 )

#             total_tensors += 1
#             total_payload_bytes += tensor_bytes
#             total_request_elapsed_s += request_elapsed_s
#             request_idx += 1
#             del tensor_cuda
#             torch.cuda.empty_cache()

#         _append_perf_log(
#             (
#                 f"[{perf_tag}] done test={test_name} model={default_model} "
#                 f"num_requests={request_idx} "
#                 f"total_tensors={total_tensors} "
#                 f"total_payload_bytes={total_payload_bytes} "
#                 f"total_request_elapsed_s={total_request_elapsed_s:.6f}"
#             )
#         )
#         _append_perf_log("")

#         # Light sanity check: verify a few sampled params were loaded as expected.
#         sample_names = list(sample_expected.keys())
#         actual = self._get_weights_by_name(
#             base_url=base_url,
#             param_names=sample_names,
#             truncate_size=5,
#             timeout=600,
#         )
#         for name in sample_names:
#             expected_values = sample_expected[name]
#             actual_values = actual.get(name)
#             assert isinstance(actual_values, list), (
#                 f"Expected list values for sampled param '{name}', got: {actual_values}"
#             )
#             assert torch.allclose(
#                 torch.tensor(actual_values),
#                 torch.tensor(expected_values),
#                 atol=2e-3,
#             ), (
#                 f"Sampled param value mismatch for '{name}'\n"
#                 f"  expected: {expected_values}\n"
#                 f"  actual:   {actual_values}"
#             )



# class TestUpdateWeightsFromTensorMultiServerNoTP(_UpdateWeightsFromTensorApiMixin):
#     """Simulate multi-engine rollout: 4 single-GPU servers (no TP), per-server updates."""

#     @pytest.fixture(scope="class", params=_ACTIVE_MODEL_PAIRS, ids=_PAIR_IDS)
#     def diffusion_servers_no_tp_4(self, request):
#         if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
#             pytest.skip("Need at least 4 CUDA GPUs for multi-server no-TP test")

#         default_model, _ = request.param
#         wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))
#         maybe_download_model(default_model)
#         _kill_existing_sglang_serve_processes()

#         base_port = get_dynamic_server_port()
#         contexts = []
#         try:
#             for i in range(4):
#                 manager = ServerManager(
#                     model=default_model,
#                     port=base_port + i,
#                     wait_deadline=wait_deadline,
#                     extra_args="--num-gpus 1",
#                     env_vars={"CUDA_VISIBLE_DEVICES": str(i)},
#                 )
#                 contexts.append(manager.start())
#             yield contexts, default_model
#         finally:
#             for ctx in reversed(contexts):
#                 ctx.cleanup()

#     def test_update_weights_from_tensor_per_server_payload_no_tp(
#         self, diffusion_servers_no_tp_4
#     ):
#         contexts, default_model = diffusion_servers_no_tp_4
#         base_urls = [f"http://localhost:{ctx.port}" for ctx in contexts]
#         perf_tag = "multi_server_no_tp_4gpu"

#         payload_by_url: dict[str, list[tuple[str, torch.Tensor]]] = {}
#         expected_values_by_url: dict[str, dict[str, list[float]]] = {}

#         # Prepare per-server payload and expected first-5 tensor values.
#         for base_url in base_urls:
#             payload = _build_direct_update(
#                 default_model,
#                 _TRANSFORMER_MODULE,
#                 value=2.0,
#                 num_params=5,
#                 max_numel=65536,
#             )
#             payload_by_url[base_url] = payload
#             expected_values_by_url[base_url] = {
#                 name: (
#                     tensor.detach()
#                     .reshape(-1)[:5]
#                     .to(device="cpu", dtype=torch.float32)
#                     .tolist()
#                 )
#                 for name, tensor in payload
#             }

#         # Fire updates concurrently: one request per independent server.
#         _append_perf_log(
#             f"[{perf_tag}] start servers={len(base_urls)} model={default_model}"
#         )
#         per_server_elapsed: dict[str, float] = {}
#         batch_t0 = time.perf_counter()

#         def _timed_update(url: str):
#             t0 = time.perf_counter()
#             result, status_code = self._update_weights_from_tensor(
#                 url,
#                 payload_by_url[url],
#                 None,
#                 [_TRANSFORMER_MODULE],
#             )
#             t1 = time.perf_counter()
#             return url, result, status_code, t1 - t0

#         with ThreadPoolExecutor(max_workers=len(base_urls)) as executor:
#             future_to_url = {
#                 executor.submit(
#                     _timed_update,
#                     base_url,
#                 ): base_url
#                 for base_url in base_urls
#             }
#             for future in as_completed(future_to_url):
#                 base_url, result, status_code, elapsed = future.result()
#                 per_server_elapsed[base_url] = elapsed
#                 assert status_code == 200, (
#                     f"Expected 200 for {base_url}, got {status_code}: {result}"
#                 )
#                 assert result.get("success", False), (
#                     f"Update failed for {base_url}: {result.get('message')}"
#                 )
#                 _append_perf_log(
#                     f"[{perf_tag}] per_server base_url={base_url} elapsed_s={elapsed:.6f} status={status_code}"
#                 )

#         batch_elapsed = time.perf_counter() - batch_t0
#         elapsed_values = list(per_server_elapsed.values())
#         _append_perf_log(
#             (
#                 f"[{perf_tag}] batch_done "
#                 f"servers={len(base_urls)} "
#                 f"batch_elapsed_s={batch_elapsed:.6f} "
#                 f"per_server_min_s={min(elapsed_values):.6f} "
#                 f"per_server_max_s={max(elapsed_values):.6f} "
#                 f"per_server_avg_s={sum(elapsed_values)/len(elapsed_values):.6f}"
#             )
#         )

#         # Validate each server independently by exact tensor values.
#         for base_url in base_urls:
#             param_names = [name for name, _ in payload_by_url[base_url]]
#             after_values = self._get_weights_by_name(
#                 base_url=base_url,
#                 param_names=param_names,
#                 truncate_size=5,
#             )
#             for name in param_names:
#                 expected = expected_values_by_url[base_url][name]
#                 actual = after_values.get(name)
#                 assert isinstance(actual, list), (
#                     f"Expected list values for '{name}' on {base_url}, got: {actual}"
#                 )
#                 assert len(actual) == len(expected), (
#                     f"Length mismatch for '{name}' on {base_url}: "
#                     f"expected {len(expected)}, got {len(actual)}"
#                 )
#                 assert torch.allclose(
#                     torch.tensor(actual),
#                     torch.tensor(expected),
#                     atol=2e-3,
#                 ), (
#                     f"Value mismatch for '{name}' on {base_url}\n"
#                     f"  expected: {expected}\n"
#                     f"  actual:   {actual}"
#                 )

#     @pytest.fixture(scope="class")
#     def qwen_image_servers_no_tp_4(self):
#         if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
#             pytest.skip("Need at least 4 CUDA GPUs for multi-GPU profiling")

#         default_model = DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST
#         wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))
#         maybe_download_model(default_model)
#         _kill_existing_sglang_serve_processes()

#         base_port = get_dynamic_server_port()
#         contexts = []
#         try:
#             for i in range(4):
#                 manager = ServerManager(
#                     model=default_model,
#                     port=base_port + i,
#                     wait_deadline=wait_deadline,
#                     extra_args="--num-gpus 1 --tp-size 1",
#                     env_vars={"CUDA_VISIBLE_DEVICES": str(i)},
#                 )
#                 contexts.append(manager.start())
#             yield contexts, default_model
#         finally:
#             for ctx in reversed(contexts):
#                 ctx.cleanup()

#     def test_profile_qwen_image_transformer_full_update_multi_gpu_no_tp(
#         self, qwen_image_servers_no_tp_4
#     ):
#         contexts, default_model = qwen_image_servers_no_tp_4
#         base_urls = [f"http://localhost:{ctx.port}" for ctx in contexts]
#         bucket_size_bytes = 4 * 1024 * 1024 * 1024
#         perf_tag = "multi_gpu_qwen_image_full_transformer_bucket4gb"
#         test_name = (
#             "TestUpdateWeightsFromTensorMultiServerNoTP."
#             "test_profile_qwen_image_transformer_full_update_multi_gpu_no_tp"
#         )

#         # Each server in this fixture is newly started with default_model already loaded.
#         # Avoid /update_weights_from_disk reset here since Qwen-Image full reload across
#         # 4 servers can exceed the default HTTP timeout and is not needed for profiling.
#         for base_url in base_urls:
#             self._get_server_info(base_url, timeout=120)

#         _append_multi_gpu_profiling_log(
#             (
#                 f"[{perf_tag}] start test={test_name} "
#                 f"model={default_model} "
#                 f"num_servers={len(base_urls)} "
#                 f"cluster_num_gpus=4 tp_size=1 "
#                 f"bucket_size_bytes={bucket_size_bytes}"
#             )
#         )

#         sample_expected: dict[str, list[float]] = {}
#         bucket_named_tensors: list[tuple[str, torch.Tensor]] = []
#         bucket_payload_bytes = 0
#         for name, tensor in _iter_module_named_tensors(default_model, _TRANSFORMER_MODULE):
#             tensor_cuda = tensor.to(device="cuda")
#             tensor_bytes = tensor_cuda.numel() * tensor_cuda.element_size()
#             if bucket_named_tensors and bucket_payload_bytes + tensor_bytes > bucket_size_bytes:
#                 pytest.skip(
#                     "Transformer payload exceeds one 4GB bucket; "
#                     "this profiling test is defined as exactly one request per server."
#                 )
#             bucket_named_tensors.append((name, tensor_cuda))
#             bucket_payload_bytes += tensor_bytes

#             if len(sample_expected) < 8:
#                 sample_expected[name] = (
#                     tensor_cuda.detach()
#                     .reshape(-1)[:5]
#                     .to(device="cpu", dtype=torch.float32)
#                     .tolist()
#                 )

#         bucket = FlattenedTensorBucket(named_tensors=bucket_named_tensors)
#         bucket_payload = {
#             _TRANSFORMER_MODULE: {
#                 "flattened_tensor": bucket.get_flattened_tensor(),
#                 "metadata": bucket.get_metadata(),
#             }
#         }
#         serialized_bucket_payload = MultiprocessingSerializer.serialize(
#             bucket_payload, output_str=True
#         )

#         per_server_elapsed_s: dict[str, float] = {}
#         first_request_sent_s: float | None = None
#         last_request_done_s = 0.0
#         time_lock = threading.Lock()

#         def _timed_server_update(url: str):
#             nonlocal first_request_sent_s, last_request_done_s
#             t_send = time.perf_counter()
#             with time_lock:
#                 if first_request_sent_s is None or t_send < first_request_sent_s:
#                     first_request_sent_s = t_send
#             result, status_code = self._update_weights_from_tensor(
#                 base_url=url,
#                 named_tensors=None,
#                 load_format="flattened_bucket",
#                 target_modules=[_TRANSFORMER_MODULE],
#                 serialized_payloads=[serialized_bucket_payload],
#                 timeout=3600,
#             )
#             t_done = time.perf_counter()
#             with time_lock:
#                 if t_done > last_request_done_s:
#                     last_request_done_s = t_done
#             return url, result, status_code, (t_done - t_send)

#         with ThreadPoolExecutor(max_workers=len(base_urls)) as executor:
#             future_to_url = {
#                 executor.submit(_timed_server_update, base_url): base_url
#                 for base_url in base_urls
#             }
#             for future in as_completed(future_to_url):
#                 base_url, result, status_code, elapsed_s = future.result()
#                 per_server_elapsed_s[base_url] = elapsed_s
#                 assert status_code == 200, (
#                     f"Expected 200 for {base_url}, got {status_code}: {result}"
#                 )
#                 assert result.get("success", False), (
#                     f"Update failed for {base_url}: {result.get('message')}"
#                 )
#                 _append_multi_gpu_profiling_log(
#                     (
#                         f"[{perf_tag}] per_server "
#                         f"base_url={base_url} "
#                         f"status={status_code} "
#                         f"request_elapsed_s={elapsed_s:.6f}"
#                     )
#                 )

#         assert first_request_sent_s is not None, "No update request was sent"
#         profiling_window_s = last_request_done_s - first_request_sent_s
#         per_server_values = list(per_server_elapsed_s.values())
#         _append_multi_gpu_profiling_log(
#             (
#                 f"[{perf_tag}] done test={test_name} "
#                 f"model={default_model} "
#                 f"num_servers={len(base_urls)} "
#                 f"bucket_payload_bytes={bucket_payload_bytes} "
#                 f"bucket_payload_gib={bucket_payload_bytes / (1024**3):.6f} "
#                 f"profiling_window_s={profiling_window_s:.6f} "
#                 f"per_server_min_s={min(per_server_values):.6f} "
#                 f"per_server_max_s={max(per_server_values):.6f} "
#                 f"per_server_avg_s={sum(per_server_values) / len(per_server_values):.6f}"
#             )
#         )
#         _append_multi_gpu_profiling_log("")

#         sample_names = list(sample_expected.keys())
#         for base_url in base_urls:
#             actual = self._get_weights_by_name(
#                 base_url=base_url,
#                 param_names=sample_names,
#                 truncate_size=5,
#                 timeout=600,
#             )
#             for name in sample_names:
#                 expected_values = sample_expected[name]
#                 actual_values = actual.get(name)
#                 assert isinstance(actual_values, list), (
#                     f"Expected list values for sampled param '{name}' on {base_url}, "
#                     f"got: {actual_values}"
#                 )
#                 assert torch.allclose(
#                     torch.tensor(actual_values),
#                     torch.tensor(expected_values),
#                     atol=2e-3,
#                 ), (
#                     f"Sampled param value mismatch for '{name}' on {base_url}\n"
#                     f"  expected: {expected_values}\n"
#                     f"  actual:   {actual_values}"
#                 )


class TestUpdateWeightsFromTensorTP(_UpdateWeightsFromTensorApiMixin):
    """E2E TP update test with per-rank payload gather via gloo."""

    @pytest.fixture(scope="class", params=_ACTIVE_MODEL_PAIRS, ids=_PAIR_IDS)
    def diffusion_server_tp(self, request):
        expected_tp_size = _get_test_tp_size()
        if expected_tp_size < 1:
            pytest.skip(f"{_TP_SIZE_ENV} must be >= 1, got {expected_tp_size}")
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() < expected_tp_size
        ):
            pytest.skip(
                f"Need at least {expected_tp_size} CUDA GPUs for TP update test"
            )

        default_model, _ = request.param
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))
        maybe_download_model(default_model)
        _kill_existing_sglang_serve_processes()

        manager = ServerManager(
            model=default_model,
            port=port,
            wait_deadline=wait_deadline,
            extra_args=f"--num-gpus {expected_tp_size} --tp-size {expected_tp_size}",
        )
        ctx = manager.start()
        try:
            yield ctx, default_model
        finally:
            ctx.cleanup()

    def test_update_weights_from_tensor_tp_gloo_payload_e2e(
        self, diffusion_server_tp
    ):
        ctx, default_model = diffusion_server_tp
        base_url = f"http://localhost:{ctx.port}"
        expected_tp_size = _get_test_tp_size()

        server_info = self._get_server_info(base_url)
        tp_size = int(server_info.get("tp_size", 1))
        if tp_size != expected_tp_size:
            pytest.skip(
                f"This test requires tp_size={expected_tp_size}, got tp_size={tp_size}"
            )

        self._reset_to_base_model(base_url, default_model)
        before_transformer_checksum = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        ).get(_TRANSFORMER_MODULE)
        assert before_transformer_checksum, "Missing pre-update transformer checksum"

        # Build rank-0 expected order first, then enforce order-based value mapping:
        # 1st->7.0, 2nd->1.0, 3rd->2.0, 4th->3.0, 5th->4.0.
        shard_candidate_min_numel = 1_000_000
        tp_value_sequence = [7.0, 1.0, 2.0, 3.0, 4.0]
        selected_params = _pick_params_from_disk(
            model_path=default_model,
            module_name=_TRANSFORMER_MODULE,
            num_params=5,
            max_numel=None,
            min_numel=shard_candidate_min_numel,
        )
        selected_param_names = [p.name for p in selected_params]
        _append_tp_debug_log(
            "[tp_debug] "
            f"tp_size={expected_tp_size} param_count={len(selected_param_names)} "
            f"param_names={selected_param_names} "
            f"value_sequence={tp_value_sequence}"
        )

        # Build TP payload list (len=tp_size). Each GPU rank serializes full tensors and rank 0
        # receives all payload strings via gloo gather_object.
        payload_session = _build_tp_payloads_via_gloo_gather(
            default_model=default_model,
            module_name=_TRANSFORMER_MODULE,
            tp_size=expected_tp_size,
            value_base=7.0,
            value_sequence=tp_value_sequence,
            num_params=5,
            max_numel=None,
            min_numel=shard_candidate_min_numel,
            param_names=selected_param_names,
        )
        expected_checksums_by_rank: list[str] | None = None
        try:
            result, status_code = self._update_weights_from_tensor(
                base_url=base_url,
                named_tensors=None,
                serialized_payloads=payload_session.serialized_payloads,
                target_modules=[_TRANSFORMER_MODULE],
                timeout=600,
            )
            assert status_code == 200, f"Expected 200, got {status_code}: {result}"
            assert result.get("success", False), (
                f"Update failed: {result.get('message')}"
            )
            expected_checksums_by_rank = _build_tp_expected_module_checksum_via_nccl(
                default_model=default_model,
                module_name=_TRANSFORMER_MODULE,
                serialized_payloads=payload_session.serialized_payloads,
                tp_size=expected_tp_size,
            )
        finally:
            payload_session.close()

        assert expected_checksums_by_rank is not None
        expected_rank0_checksum = expected_checksums_by_rank[0]
        after_transformer_checksum = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        ).get(_TRANSFORMER_MODULE)
        _append_tp_debug_log(
            "[tp_debug] "
            f"tp_size={expected_tp_size} "
            f"before_checksum={before_transformer_checksum} "
            f"after_checksum={after_transformer_checksum} "
            f"expected_rank0_checksum={expected_rank0_checksum}"
        )
        assert after_transformer_checksum != before_transformer_checksum, (
            f"Expected TP tensor update to change rank-0 transformer checksum "
            f"(before={before_transformer_checksum}, after={after_transformer_checksum})."
        )
        assert after_transformer_checksum == expected_rank0_checksum, (
            f"TP={expected_tp_size} rank-0 transformer checksum mismatch\n"
            f"  expected(rank0): {expected_rank0_checksum}\n"
            f"  actual(rank0):   {after_transformer_checksum}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
