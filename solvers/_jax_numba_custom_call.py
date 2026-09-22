"""JAX custom-call bridge for launching numba-cuda kernels.

The public surface in this module is intentionally small: compile a CUDA
kernel, register a typed XLA FFI launcher for it, and call it from JAX.  The C++ FFI shim is built lazily into ``/tmp`` so the project can
keep using plain ``uv run python`` without a package build step.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

_CAPSULE_NAME = b"xla._CUSTOM_CALL_TARGET"
_TARGET_NAME = "menodax_numba_cuda_abi_launch"
_CUSTOM_CALL_API_VERSION = 4
_REGISTERED = False
_LOADED_LIB: ctypes.CDLL | None = None

ABI_ARRAY = 0
ABI_SCALAR_F64 = 1
ABI_SCALAR_I32 = 2


@dataclass(frozen=True)
class CudaLaunch:
    """Compiled CUDA kernel launch metadata for an XLA FFI call."""

    function: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_mem: int = 0


def _as_3d(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), 1, 1)
    parts = tuple(int(v) for v in value)
    if len(parts) == 1:
        return (parts[0], 1, 1)
    if len(parts) == 2:
        return (parts[0], parts[1], 1)
    if len(parts) == 3:
        return parts
    raise ValueError(f"launch dimensions must have rank 1, 2, or 3; got {value!r}")


def _pycapsule_new(ptr: int, name: bytes = _CAPSULE_NAME) -> object:
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    return ctypes.pythonapi.PyCapsule_New(ctypes.c_void_p(ptr), name, None)


def _source() -> str:
    return r"""
#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

using CuLaunchKernel = int (*)(void*, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, void*, void**, void**);

static CuLaunchKernel LoadCuLaunchKernel() {
  static std::once_flag once;
  static CuLaunchKernel fn = nullptr;
  std::call_once(once, []() {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return;
    fn = reinterpret_cast<CuLaunchKernel>(dlsym(lib, "cuLaunchKernel"));
  });
  return fn;
}

// numba-cuda-mlir lowers an array parameter to an MLIR MemRef descriptor,
// passed flattened as {allocated, aligned, offset, sizes..., strides...} --
// 3 + 2*rank kernel parameters.  The offset and strides count ELEMENTS, unlike
// the byte strides of the numba-cuda array ABI this replaced (which also led
// with a meminfo/parent pair, for 5 + 2*rank parameters).
struct ArrayArg {
  void* allocated = nullptr;
  void* aligned = nullptr;
  int64_t offset = 0;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct KernelArgStorage {
  ArrayArg array;
  double f64 = 0.0;
  int32_t i32 = 0;
};

static void AddArrayParams(ffi::AnyBuffer buf, KernelArgStorage& storage,
                           std::vector<void*>& params) {
  storage.array.allocated = buf.untyped_data();
  storage.array.aligned = buf.untyped_data();
  storage.array.offset = 0;
  auto dims = buf.dimensions();
  storage.array.sizes.assign(dims.begin(), dims.end());
  storage.array.strides.resize(storage.array.sizes.size());
  // Row-major element counts, so the innermost stride is 1.
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(storage.array.sizes.size()) - 1; i >= 0; --i) {
    storage.array.strides[static_cast<size_t>(i)] = stride;
    stride *= storage.array.sizes[static_cast<size_t>(i)];
  }

  params.push_back(&storage.array.allocated);
  params.push_back(&storage.array.aligned);
  params.push_back(&storage.array.offset);
  for (int64_t& size : storage.array.sizes) params.push_back(&size);
  for (int64_t& stride_value : storage.array.strides) params.push_back(&stride_value);
}

static ffi::Error AddBufferParam(ffi::AnyBuffer buf, int64_t kind,
                                 KernelArgStorage& storage,
                                 std::vector<void*>& params) {
  if (kind != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "unknown Numba CUDA ABI argument kind");
  }
  AddArrayParams(buf, storage, params);
  return ffi::Error::Success();
}

static ffi::Error LaunchNumbaCudaAbi(
    void* stream, int64_t function, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t shared_mem, ffi::Span<const int64_t> arg_kinds,
    ffi::Span<const double> scalar_f64_values,
    ffi::Span<const int32_t> scalar_i32_values,
    ffi::RemainingArgs args, ffi::RemainingRets rets) {
  CuLaunchKernel cuLaunchKernel = LoadCuLaunchKernel();
  if (cuLaunchKernel == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "could not load cuLaunchKernel from libcuda.so.1");
  }
  std::vector<KernelArgStorage> storage(arg_kinds.size());
  std::vector<void*> params;
  params.reserve(arg_kinds.size() * 12);
  size_t arg_idx = 0;
  size_t ret_idx = 0;
  size_t f64_idx = 0;
  size_t i32_idx = 0;

  for (size_t i = 0; i < arg_kinds.size(); ++i) {
    const int64_t kind = arg_kinds[i];
    if (kind == 1) {
      if (f64_idx >= scalar_f64_values.size()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "not enough f64 scalar values");
      }
      storage[i].f64 = scalar_f64_values[f64_idx++];
      params.push_back(&storage[i].f64);
    } else if (kind == 2) {
      if (i32_idx >= scalar_i32_values.size()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "not enough i32 scalar values");
      }
      storage[i].i32 = scalar_i32_values[i32_idx++];
      params.push_back(&storage[i].i32);
    } else if (arg_idx < args.size()) {
      auto arg = args.get<ffi::AnyBuffer>(arg_idx++);
      if (!arg.has_value()) return arg.error();
      ffi::Error err = AddBufferParam(arg.value(), kind, storage[i], params);
      if (!err.success()) return err;
    } else {
      auto ret = rets.get<ffi::AnyBuffer>(ret_idx++);
      if (!ret.has_value()) return ret.error();
      ffi::Error err = AddBufferParam(*ret.value(), kind, storage[i], params);
      if (!err.success()) return err;
    }
  }
  if (arg_idx != args.size() || ret_idx != rets.size()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "kernel ABI kinds did not consume all buffers");
  }

  int err = cuLaunchKernel(reinterpret_cast<void*>(function),
                           static_cast<unsigned int>(grid_x),
                           static_cast<unsigned int>(grid_y),
                           static_cast<unsigned int>(grid_z),
                           static_cast<unsigned int>(block_x),
                           static_cast<unsigned int>(block_y),
                           static_cast<unsigned int>(block_z),
                           static_cast<unsigned int>(shared_mem),
                           stream, params.data(), nullptr);
  if (err != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "cuLaunchKernel failed with CUDA driver error " +
                          std::to_string(err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    menodax_numba_cuda_abi_launch, LaunchNumbaCudaAbi,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .Attr<int64_t>("function")
        .Attr<int64_t>("grid_x")
        .Attr<int64_t>("grid_y")
        .Attr<int64_t>("grid_z")
        .Attr<int64_t>("block_x")
        .Attr<int64_t>("block_y")
        .Attr<int64_t>("block_z")
        .Attr<int64_t>("shared_mem")
        .Attr<ffi::Span<const int64_t>>("arg_kinds")
        .Attr<ffi::Span<const double>>("scalar_f64_values")
        .Attr<ffi::Span<const int32_t>>("scalar_i32_values")
        .RemainingArgs()
        .RemainingRets());
"""


def _build_bridge() -> Path:
    include_dir = Path(jax.ffi.include_dir())
    build_dir = Path(tempfile.gettempdir()) / "menodax_jax_numba_cuda_bridge"
    build_dir.mkdir(parents=True, exist_ok=True)
    source = _source()
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    src_path = build_dir / f"bridge-{digest}.cc"
    so_path = build_dir / f"bridge-{digest}{sysconfig.get_config_var('EXT_SUFFIX')}"
    if so_path.exists():
        return so_path
    src_path.write_text(source)
    cmd = [
        "g++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        f"-I{include_dir}",
        str(src_path),
        "-ldl",
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return so_path


def register_target() -> None:
    """Register the generic CUDA launcher with JAX once per process."""

    global _LOADED_LIB, _REGISTERED
    if _REGISTERED:
        return
    so_path = _build_bridge()
    _LOADED_LIB = ctypes.CDLL(str(so_path))
    symbol = getattr(_LOADED_LIB, _TARGET_NAME)
    capsule = _pycapsule_new(ctypes.cast(symbol, ctypes.c_void_p).value)
    jax.ffi.register_ffi_target(_TARGET_NAME, capsule, platform="CUDA", api_version=1)
    _REGISTERED = True


# Loaded modules are held for the process lifetime: the ``CUfunction`` handed to
# the FFI target stays valid only while its module is loaded, and these kernels
# live as long as the JAX primitives that launch them.
_LOADED_MODULES: list[Any] = []


def compile_kernel(kernel: Any, argtypes: Sequence[Any]) -> int:
    """Compile a ``cuda.jit`` kernel and return its ``CUfunction`` pointer.

    numba-cuda-mlir exposes no ``get_cufunc()`` -- its ``MLIRLibrary`` only
    offers the textual IR.  The linked cubin and the mangled entry name are on
    the compile result's metadata instead, so load the module through the driver
    and look the function up by name.
    """

    cres = kernel.compile(tuple(argtypes)).cres
    cubin = cres.metadata["cubin"]
    func_name = cres.metadata["func_name"]

    libcuda = ctypes.CDLL("libcuda.so.1")
    module = ctypes.c_void_p()
    err = libcuda.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(cubin))
    if err != 0:
        raise RuntimeError(f"cuModuleLoadData failed with CUDA driver error {err}")
    _LOADED_MODULES.append(module)

    function = ctypes.c_void_p()
    err = libcuda.cuModuleGetFunction(
        ctypes.byref(function), module, func_name.encode()
    )
    if err != 0:
        raise RuntimeError(f"cuModuleGetFunction failed with CUDA driver error {err}")
    if function.value is None:
        raise RuntimeError("cuModuleGetFunction returned a null function pointer")
    return int(function.value)


def make_launch(
    kernel: Any,
    argtypes: Sequence[Any],
    *,
    grid: int | Sequence[int],
    block: int | Sequence[int],
    shared_mem: int = 0,
) -> CudaLaunch:
    return CudaLaunch(
        function=compile_kernel(kernel, argtypes),
        grid=_as_3d(grid),
        block=_as_3d(block),
        shared_mem=int(shared_mem),
    )


def ffi_abi_call(
    launch: CudaLaunch,
    inputs: Sequence[Any],
    output_specs: Sequence[jax.ShapeDtypeStruct],
    *,
    input_kinds: Sequence[int],
    scalar_f64_values: Sequence[float] = (),
    scalar_i32_values: Sequence[int] = (),
) -> tuple[Any, ...]:
    """Launch a Numba CUDA kernel using Numba's normal array/scalar ABI.

    Outputs are always arrays, so only the input kinds need spelling out.
    """

    register_target()
    attrs = {
        "function": np.int64(launch.function),
        "grid_x": np.int64(launch.grid[0]),
        "grid_y": np.int64(launch.grid[1]),
        "grid_z": np.int64(launch.grid[2]),
        "block_x": np.int64(launch.block[0]),
        "block_y": np.int64(launch.block[1]),
        "block_z": np.int64(launch.block[2]),
        "shared_mem": np.int64(launch.shared_mem),
        "arg_kinds": np.asarray(
            tuple(input_kinds) + (ABI_ARRAY,) * len(output_specs), dtype=np.int64
        ),
        "scalar_f64_values": np.asarray(tuple(scalar_f64_values), dtype=np.float64),
        "scalar_i32_values": np.asarray(tuple(scalar_i32_values), dtype=np.int32),
    }
    result = jax.ffi.ffi_call(
        _TARGET_NAME,
        tuple(output_specs),
        has_side_effect=False,
        custom_call_api_version=_CUSTOM_CALL_API_VERSION,
    )(*inputs, **attrs)
    if not isinstance(result, tuple):
        return (result,)
    return result
