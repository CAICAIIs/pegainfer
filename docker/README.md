# OpenInfer development container

The development image contains the native toolchain required to build
OpenInfer: CUDA, the Rust nightly pinned by `rust-toolchain.toml`, Python 3,
uv, clang, OpenSSL, protoc, and NCCL 2.30.4 or newer.

Build it once:

```bash
docker/dev.sh build
```

Open a shell with the repository at its original absolute path and persistent
compiler caches mounted:

```bash
docker/dev.sh shell
```

Linked Git worktrees are supported: the wrapper detects an external Git
common directory and mounts it read-only so build scripts can inspect
submodule state.

Or run a one-off build:

```bash
docker/dev.sh run cargo build --release
```

Mount model weights read-only at their existing absolute path:

```bash
OPENINFER_MODEL_DIR=/models/Qwen3-4B docker/dev.sh shell
```

The default base is the pinned CUDA 13.2 development image, which is the
newest version supported by the current GB300 tray driver. Override it without
changing the Dockerfile after upgrading the host driver:

```bash
CUDA_IMAGE=nvidia/cuda:<version>-devel-ubuntu24.04 docker/dev.sh build
```

On a tray without a GIN-capable NIC, pass `EP_DISABLE_GIN=1` when starting
the container. It is intentionally not baked into the image because networked
deployments require GIN.
