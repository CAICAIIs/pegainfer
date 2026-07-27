#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${OPENINFER_DEV_IMAGE:-openinfer-dev:cu132}"
container="${OPENINFER_DEV_CONTAINER:-openinfer-dev}"
cache_root="${OPENINFER_DEV_CACHE:-$HOME/.cache/openinfer-dev}"

usage() {
  cat <<'EOF'
Usage:
  docker/dev.sh build             Build the development image.
  docker/dev.sh shell [COMMAND]   Start an interactive development container.
  docker/dev.sh run COMMAND...    Run a command in a disposable container.

Environment:
  CUDA_IMAGE              CUDA devel base (default: CUDA 13.2 / Ubuntu 24.04).
  OPENINFER_DEV_IMAGE     Image tag (default: openinfer-dev:cu132).
  OPENINFER_DEV_CONTAINER Interactive container name (default: openinfer-dev).
  OPENINFER_DEV_CACHE     Persistent build-cache directory.
  EP_DISABLE_GIN          Forwarded when set; useful on trays without a GIN NIC.
EOF
}

build_image() {
  docker build \
    --file "$repo_root/docker/Dockerfile.dev" \
    --build-arg "CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:13.2.0-devel-ubuntu24.04}" \
    --build-arg "DEV_USER=$(id -un)" \
    --build-arg "DEV_UID=$(id -u)" \
    --build-arg "DEV_GID=$(id -g)" \
    --tag "$image" \
    "$repo_root"
}

docker_args=(
  --gpus all
  --ipc host
  --network host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --volume "$repo_root:/workspace/openinfer"
  --volume "$cache_root/cargo-registry:/opt/cargo/registry"
  --volume "$cache_root/cargo-git:/opt/cargo/git"
  --volume "$cache_root/target:/workspace/openinfer/target"
  --workdir /workspace/openinfer
)

if [[ -n "${EP_DISABLE_GIN:-}" ]]; then
  docker_args+=(--env "EP_DISABLE_GIN=$EP_DISABLE_GIN")
fi

case "${1:-}" in
  build)
    build_image
    ;;
  shell)
    shift
    mkdir -p "$cache_root"/{cargo-registry,cargo-git,target}
    if (( $# == 0 )); then
      set -- /bin/bash
    fi
    docker run --rm -it --name "$container" "${docker_args[@]}" "$image" "$@"
    ;;
  run)
    shift
    (( $# > 0 )) || { usage >&2; exit 2; }
    mkdir -p "$cache_root"/{cargo-registry,cargo-git,target}
    docker run --rm "${docker_args[@]}" "$image" "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
