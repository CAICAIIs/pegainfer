#!/usr/bin/env bash
set -euo pipefail

# Fixed-tray GLM5.2 native-MTP P/D development stack:
#   tray03: PegaFlow MetaServer + TP4 prefill + vLLM Router
#   tray04: EP4 decode
#
# Every value can be overridden from the environment so the same script can
# be reused after the fixed trays are reassigned.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG_FILE=${GLM52_PD_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/openinfer/glm52-pd.env}
if [[ -f $CONFIG_FILE ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

required_vars=(
    P_HOST
    D_HOST
    P_IMAGE
    D_IMAGE
    P_MODEL_PATH
    D_MODEL_PATH
    ROUTER_BIN
    OPENINFER_NCCL_ROOT
)
for var in "${required_vars[@]}"; do
    if [[ -z ${!var:-} ]]; then
        printf '%s must be set in the environment or %s\n' "$var" "$CONFIG_FILE" >&2
        exit 2
    fi
done

P_CONTAINER=${P_CONTAINER:-openinfer-pd-prefill}
D_CONTAINER=${D_CONTAINER:-openinfer-pd-decode}
P_HTTP_PORT=${P_HTTP_PORT:-8000}
D_HTTP_PORT=${D_HTTP_PORT:-8000}
ROUTER_PORT=${ROUTER_PORT:-10001}
METASERVER_PORT=${METASERVER_PORT:-50056}
METASERVER_HTTP_PORT=${METASERVER_HTTP_PORT:-19092}
P_TRANSFER_PORT=${P_TRANSFER_PORT:-50103}
D_TRANSFER_PORT=${D_TRANSFER_PORT:-50104}
RDMA_NIC=${RDMA_NIC:-mlx5_bond_0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
KV_OFFLOAD_HOST_GIB=${KV_OFFLOAD_HOST_GIB:-8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5.2-fp8}

HOST_REPO=${HOST_REPO:-$REPO_ROOT}
P_REPO=${P_REPO:-/workspace/openinfer}
D_REPO=${D_REPO:-/workspace/openinfer}

discover_ip() {
    ssh "$1" "ip -o -4 addr show dev $RDMA_NIC | awk 'NR == 1 { split(\\$4, a, \"/\"); print a[1] }'"
}

P_IP=${P_IP:-$(discover_ip "$P_HOST")}
D_IP=${D_IP:-$(discover_ip "$D_HOST")}
if [[ -z $P_IP || -z $D_IP ]]; then
    printf 'could not discover an IPv4 address on %s; set P_IP and D_IP explicitly\n' \
        "$RDMA_NIC" >&2
    exit 2
fi

role_pid_file() {
    printf '/tmp/openinfer-glm52-pd-%s.pid' "$1"
}

role_log_file() {
    printf '/tmp/openinfer-glm52-pd-%s.log' "$1"
}

shell_join() {
    local joined
    printf -v joined '%q ' "$@"
    printf '%s' "$joined"
}

container_start() {
    local host=$1 container=$2 role=$3 command=$4
    local pid_file log_file wrapped quoted
    pid_file=$(role_pid_file "$role")
    log_file=$(role_log_file "$role")
    container_stop "$host" "$container" "$role"
    printf -v wrapped 'echo $$ > %q; %s > %q 2>&1' \
        "$pid_file" "$command" "$log_file"
    printf -v quoted '%q' "$wrapped"
    ssh "$host" "docker exec -d $container bash -lc $quoted"
}

container_stop() {
    local host=$1 container=$2 role=$3
    local pid_file pid
    pid_file=$(role_pid_file "$role")
    pid=$(ssh "$host" "docker exec $container bash -lc 'test -f $pid_file && cat $pid_file'" \
        2>/dev/null || true)
    if [[ $pid =~ ^[0-9]+$ ]]; then
        ssh "$host" "docker exec $container kill -TERM $pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            if ! ssh "$host" "docker exec $container kill -0 $pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
    fi
    ssh "$host" "docker exec $container rm -f $pid_file" 2>/dev/null || true
}

host_start() {
    local host=$1 role=$2 command=$3
    local pid_file log_file wrapped quoted
    pid_file=$(role_pid_file "$role")
    log_file=$(role_log_file "$role")
    host_stop "$host" "$role"
    printf -v wrapped 'echo $$ > %q; %s > %q 2>&1' \
        "$pid_file" "$command" "$log_file"
    printf -v quoted '%q' "$wrapped"
    ssh "$host" "nohup bash -lc $quoted >/dev/null 2>&1 &"
}

host_stop() {
    local host=$1 role=$2
    local pid_file pid
    pid_file=$(role_pid_file "$role")
    pid=$(ssh "$host" "test -f $pid_file && cat $pid_file" 2>/dev/null || true)
    if [[ $pid =~ ^[0-9]+$ ]]; then
        ssh "$host" "kill -TERM $pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            if ! ssh "$host" "kill -0 $pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
    fi
    ssh "$host" "rm -f $pid_file" 2>/dev/null || true
}

wait_http() {
    local name=$1 url=$2 timeout=${3:-600}
    local start=$SECONDS
    until curl -fsS --max-time 2 "$url" >/dev/null 2>&1; do
        if (( SECONDS - start >= timeout )); then
            printf '%s did not become ready within %ss: %s\n' "$name" "$timeout" "$url" >&2
            return 1
        fi
        sleep 2
    done
    printf '%s ready: %s\n' "$name" "$url"
}

ensure_containers() {
    if ! ssh "$P_HOST" "docker inspect $P_CONTAINER >/dev/null 2>&1"; then
        ssh "$P_HOST" "docker run -d \
            --name $P_CONTAINER \
            --gpus all \
            --network host \
            --ipc host \
            --security-opt label=disable \
            --ulimit memlock=-1:-1 \
            --device /dev/infiniband/uverbs4 \
            --device /dev/infiniband/rdma_cm \
            -v /dev/infiniband:/dev/infiniband \
            -v $HOST_REPO:$P_REPO \
            -v $P_MODEL_PATH:$P_MODEL_PATH:ro \
            --entrypoint bash \
            $P_IMAGE -lc 'sleep infinity' >/dev/null"
    else
        ssh "$P_HOST" "docker start $P_CONTAINER >/dev/null"
    fi
    if ! ssh "$D_HOST" "docker inspect $D_CONTAINER >/dev/null 2>&1"; then
        ssh "$D_HOST" "docker run -d \
            --name $D_CONTAINER \
            --gpus all \
            --network host \
            --ipc host \
            --security-opt label=disable \
            --ulimit memlock=-1:-1 \
            --device /dev/infiniband/uverbs4 \
            --device /dev/infiniband/rdma_cm \
            -v /dev/infiniband:/dev/infiniband \
            -v $HOST_REPO:$D_REPO \
            -v $D_MODEL_PATH:$D_MODEL_PATH:ro \
            --entrypoint bash \
            $D_IMAGE -lc 'sleep infinity' >/dev/null"
    else
        ssh "$D_HOST" "docker start $D_CONTAINER >/dev/null"
    fi

    ensure_nccl "$P_HOST" "$P_CONTAINER"
    ensure_nccl "$D_HOST" "$D_CONTAINER"

    ssh "$P_HOST" "docker exec $P_CONTAINER bash -lc \
        'nvidia-smi -L >/dev/null && test -c /dev/infiniband/uverbs4'"
    ssh "$D_HOST" "docker exec $D_CONTAINER bash -lc \
        'nvidia-smi -L >/dev/null && test -c /dev/infiniband/uverbs4'"
}

ensure_nccl() {
    local host=$1 container=$2
    if ssh "$host" "docker exec $container bash -lc \
        'nm -D /usr/lib/aarch64-linux-gnu/libnccl.so.2 | grep -q ncclCommQueryProperties'"; then
        return
    fi
    printf 'Upgrading NCCL in %s to 2.30.7...\n' "$container"
    ssh "$host" "docker exec $container bash -lc \
        'apt-get update -qq && \
         apt-get install -y -qq \
            libnccl2=2.30.7-1+cuda13.3 \
            libnccl-dev=2.30.7-1+cuda13.3'"
}

prepare() {
    local pegaflow_manifest
    ensure_containers
    printf 'Building OpenInfer GLM5.2 release binary on %s...\n' "$P_HOST"
    ssh "$P_HOST" "docker exec \
        -e OPENINFER_NCCL_ROOT=$OPENINFER_NCCL_ROOT \
        -e OPENINFER_CUDA_SM=103 \
        $P_CONTAINER bash -lc \
        'cd $P_REPO && cargo build --release --no-default-features --features glm52'"

    pegaflow_manifest=$(ssh "$P_HOST" "docker exec $P_CONTAINER bash -lc \
        'find /root/.cargo/git/checkouts/pegaflow-* -path \"*/1473c53/pegaflow-metaserver/Cargo.toml\" -print -quit'")
    if [[ -z $pegaflow_manifest ]]; then
        printf 'pegaflow 1473c53 checkout is missing in %s\n' "$P_CONTAINER" >&2
        return 1
    fi
    printf 'Building PegaFlow MetaServer...\n'
    ssh "$P_HOST" "docker exec \
        -e CARGO_TARGET_DIR=$P_REPO/target/pegaflow \
        $P_CONTAINER bash -lc \
        'cargo build --release --manifest-path $pegaflow_manifest'"

    ssh "$D_HOST" "docker exec $D_CONTAINER bash -lc \
        'nm -D /usr/lib/aarch64-linux-gnu/libnccl.so.2 | grep -q ncclCommQueryProperties'"
    ssh "$D_HOST" "docker exec $D_CONTAINER test -x $D_REPO/target/release/openinfer"
    ssh "$P_HOST" "test -x $ROUTER_BIN"
    printf 'prepare complete\n'
}

start() {
    local meta_cmd p_cmd d_cmd router_cmd
    local common_p2p
    common_p2p="--kv-offload --kv-offload-host-gib $KV_OFFLOAD_HOST_GIB \
--kv-p2p-metaserver-addr http://$P_IP:$METASERVER_PORT \
--kv-p2p-nics $RDMA_NIC"

    meta_cmd="exec $(shell_join \
        "$P_REPO/target/pegaflow/release/pegaflow-metaserver" \
        --addr "0.0.0.0:$METASERVER_PORT" \
        --http-addr "0.0.0.0:$METASERVER_HTTP_PORT" \
        --log-level info)"
    container_start "$P_HOST" "$P_CONTAINER" meta "$meta_cmd"
    wait_http metaserver "http://$P_IP:$METASERVER_HTTP_PORT/health" 30

    p_cmd="cd $(printf %q "$P_REPO") && exec env RUST_LOG=info \
EP_DISABLE_GIN=1 NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32 \
$(printf %q "$P_REPO/target/release/openinfer") \
--model-path $(printf %q "$P_MODEL_PATH") \
--served-model-name $(printf %q "$SERVED_MODEL_NAME") \
--port $P_HTTP_PORT --tp-size 4 --moe-topo tp4 \
--glm52-prefill-only --glm52-native-mtp \
--glm52-weight-staging --max-model-len $MAX_MODEL_LEN \
$common_p2p --kv-p2p-advertise-addr $P_IP:$P_TRANSFER_PORT"
    d_cmd="cd $(printf %q "$D_REPO") && exec env RUST_LOG=info \
EP_DISABLE_GIN=1 \
$(printf %q "$D_REPO/target/release/openinfer") \
--model-path $(printf %q "$D_MODEL_PATH") \
--served-model-name $(printf %q "$SERVED_MODEL_NAME") \
--port $D_HTTP_PORT --moe-topo ep4 \
--glm52-native-mtp --glm52-weight-staging \
--max-model-len $MAX_MODEL_LEN \
$common_p2p --kv-p2p-advertise-addr $D_IP:$D_TRANSFER_PORT"

    container_start "$P_HOST" "$P_CONTAINER" prefill "$p_cmd"
    container_start "$D_HOST" "$D_CONTAINER" decode "$d_cmd"
    wait_http prefill "http://$P_IP:$P_HTTP_PORT/health" 600
    wait_http decode "http://$D_IP:$D_HTTP_PORT/health" 600

    router_cmd="exec $(shell_join \
        "$ROUTER_BIN" \
        --host 0.0.0.0 \
        --port "$ROUTER_PORT" \
        --vllm-pd-disaggregation \
        --prefill "http://$P_IP:$P_HTTP_PORT" none \
        --decode "http://$D_IP:$D_HTTP_PORT" \
        --prefill-policy round_robin \
        --decode-policy round_robin \
        --intra-node-data-parallel-size 1 \
        --kv-connector nixl \
        --disable-retries \
        --prometheus-port 29001)"
    host_start "$P_HOST" router "$router_cmd"
    wait_http router "http://$P_IP:$ROUTER_PORT/health" 30
}

stop() {
    host_stop "$P_HOST" router
    container_stop "$D_HOST" "$D_CONTAINER" decode
    container_stop "$P_HOST" "$P_CONTAINER" prefill
    container_stop "$P_HOST" "$P_CONTAINER" meta
}

status_one() {
    local host=$1 where=$2 role=$3 pid_file pid
    pid_file=$(role_pid_file "$role")
    if [[ $where == host ]]; then
        pid=$(ssh "$host" "test -f $pid_file && cat $pid_file" 2>/dev/null || true)
        if [[ $pid =~ ^[0-9]+$ ]] && ssh "$host" "kill -0 $pid" 2>/dev/null; then
            printf '%-10s running pid=%s host=%s\n' "$role" "$pid" "$host"
        else
            printf '%-10s stopped host=%s\n' "$role" "$host"
        fi
    else
        pid=$(ssh "$host" "docker exec $where bash -lc 'test -f $pid_file && cat $pid_file'" \
            2>/dev/null || true)
        if [[ $pid =~ ^[0-9]+$ ]] \
            && ssh "$host" "docker exec $where kill -0 $pid" 2>/dev/null; then
            printf '%-10s running pid=%s host=%s container=%s\n' "$role" "$pid" "$host" "$where"
        else
            printf '%-10s stopped host=%s container=%s\n' "$role" "$host" "$where"
        fi
    fi
}

status() {
    status_one "$P_HOST" "$P_CONTAINER" meta
    status_one "$P_HOST" "$P_CONTAINER" prefill
    status_one "$D_HOST" "$D_CONTAINER" decode
    status_one "$P_HOST" host router
}

logs() {
    local role=${1:-all}
    case "$role" in
        meta|prefill)
            ssh "$P_HOST" "docker exec $P_CONTAINER tail -n 120 $(role_log_file "$role")"
            ;;
        decode)
            ssh "$D_HOST" "docker exec $D_CONTAINER tail -n 120 $(role_log_file "$role")"
            ;;
        router)
            ssh "$P_HOST" "tail -n 120 $(role_log_file "$role")"
            ;;
        all)
            for item in meta prefill decode router; do
                printf '\n===== %s =====\n' "$item"
                logs "$item" || true
            done
            ;;
        *)
            printf 'unknown role: %s\n' "$role" >&2
            return 2
            ;;
    esac
}

smoke() {
    python3 "$(dirname "$0")/glm52_pd_smoke.py" \
        --base-url "http://$P_IP:$ROUTER_PORT" \
        --model "$SERVED_MODEL_NAME"
}

usage() {
    printf 'usage: %s {prepare|start|stop|restart|status|logs [role]|smoke}\n' "$0"
}

case "${1:-}" in
    prepare) prepare ;;
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    logs) logs "${2:-all}" ;;
    smoke) smoke ;;
    *) usage; exit 2 ;;
esac
