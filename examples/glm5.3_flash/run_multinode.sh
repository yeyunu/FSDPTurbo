#!/bin/bash
# ============================================================
# GLM5.3 flash Multi-node Distributed Training Launch Script
#
# Static rank assignment via --node_rank.
#
# HOSTS (defined below) lists all node IPs, master first; each node
# infers its NODE_RANK from its own IP's position in the list, so the
# same command works on every node:
#   bash run_multinode.sh
# Override when needed:
#   NODE_RANK=1 ...        explicit rank, skips inference
#   HOSTS=a,b,c ...        different cluster
#   NNODES=3 ...           only when running without HOSTS
# ============================================================

set -e

ulimit -n 65535

export HCCL_SOCKET_IFNAME=enp48s3u1u1
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export TP_SOCKET_IFNAME=enp48s3u1u1
export HCCL_CONNECT_TIMEOUT=1200

export ASCEND_COREDUMP_SIGNAL=none
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

# --- Cluster Configuration ---
# Master node IP, must be set manually and identical on every node.
MASTER_ADDR=${MASTER_ADDR:-"*.*.*.*"}
if [[ -z "$MASTER_ADDR" ]]; then
    echo "[ERROR] MASTER_ADDR is not set. Example: MASTER_ADDR=10.0.0.1 NNODES=2 bash $0" >&2
    exit 1
fi

# Total number of nodes, identical on every node. Defaults to the
# number of entries in HOSTS (resolved below); set explicitly only
# when running without HOSTS.
NNODES=${NNODES:-""}

MASTER_PORT=${MASTER_PORT:-29500}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}

# All node IPs of this job, comma-separated, MASTER FIRST. The position
# of this machine's IP in the list is its NODE_RANK. Edit here when the
# cluster changes; can also be overridden from the command line.
HOSTS=${HOSTS:-"*.*.*.*,*.*.*.*"}
NNODES=${NNODES:-$(echo "$HOSTS" | awk -F',' '{print NF}')}
if [[ -z "$NNODES" ]]; then
    echo "[ERROR] NNODES could not be resolved. Set NNODES or HOSTS." >&2
    exit 1
fi

# --- Training Arguments ---
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------
# Detect every IPv4 address of the current machine (a node may
# have several NICs, so we match the master against all of them).
# ------------------------------------------------------------
get_local_ips() {
    local ips=""
    # hostname -I: space-separated list of all addresses
    ips=$(hostname -I 2>/dev/null || true)
    if [[ -z "$ips" ]]; then
        # fallback: enumerate global-scope IPv4 addresses
        ips=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
    fi
    echo "$ips"
}

LOCAL_IPS=$(get_local_ips)

is_master() {
    local ip
    for ip in $LOCAL_IPS; do
        if [[ "$ip" == "$MASTER_ADDR" ]]; then
            return 0
        fi
    done
    return 1
}

# ------------------------------------------------------------
# Resolve NODE_RANK:
#   master (by IP match)          -> 0
#   NODE_RANK env already set     -> use it as-is
#   HOSTS given                   -> position of this machine's IP
#   otherwise                     -> fatal error with guidance
# ------------------------------------------------------------
if [[ -n "${NODE_RANK:-}" ]]; then
    # Explicit override wins.
    :
elif is_master; then
    NODE_RANK=0
elif [[ -n "${HOSTS:-}" ]]; then
    # HOSTS: comma-separated node IPs, MASTER first. No spaces.
    NODE_RANK=""
    idx=0
    IFS=',' read -ra HOST_LIST <<< "$HOSTS"
    for host in "${HOST_LIST[@]}"; do
        for ip in $LOCAL_IPS; do
            if [[ "$ip" == "$host" ]]; then
                NODE_RANK=$idx
                break 2
            fi
        done
        idx=$((idx + 1))
    done
    if [[ -z "$NODE_RANK" ]]; then
        echo "[ERROR] None of this machine's IPs ($LOCAL_IPS) is listed in HOSTS ($HOSTS)." >&2
        echo "        Check HOSTS or set NODE_RANK explicitly." >&2
        exit 1
    fi
else
    echo "[ERROR] This machine is a worker (local IPs: $LOCAL_IPS, master: $MASTER_ADDR)" >&2
    echo "        but NODE_RANK is not set and HOSTS is not provided, so its rank" >&2
    echo "        cannot be inferred. Either:" >&2
    echo "          NODE_RANK=1 bash $0" >&2
    echo "        or on ALL nodes:" >&2
    echo "          HOSTS=10.0.0.1,10.0.0.2,... bash $0" >&2
    exit 1
fi

if [[ "$NODE_RANK" -ge "$NNODES" ]]; then
    echo "[ERROR] NODE_RANK=$NODE_RANK is out of range for NNODES=$NNODES." >&2
    exit 1
fi

ROLE="master"
if [[ "$NODE_RANK" -ne 0 ]]; then
    ROLE="worker(node $NODE_RANK)"
fi

# Model reduction (layers/experts) is a standalone offline step now -- run
# python examples/glm5.3_flash/convert_checkpoint.py once (pass --num-hidden-layers
# / --n-routed-experts explicitly; no defaults), then point
# config.yaml's model.model_name_or_path at the reduced checkpoint.

mkdir -p "$SCRIPT_DIR/log"
LOGFILE="$SCRIPT_DIR/log/train_glm53flash.log"

echo "============================================================"
echo "  glm5.3-flash Multi-node Training"
echo "  ROLE:          $ROLE"
echo "  NODE_RANK:     $NODE_RANK"
echo "  LOCAL_IPS:     ${LOCAL_IPS//$'\n'/ }"
echo "  MASTER_ADDR:   $MASTER_ADDR"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "  NNODES:        $NNODES"
echo "  NPUS_PER_NODE: $NPUS_PER_NODE"
echo "  CONFIG:        $CONFIG_FILE"
echo "  LOG:           $LOGFILE"
echo "============================================================"

# --- CANN Environment ---
# CANN_DIR=${CANN_DIR:-/usr/local/Ascend}
# source $CANN_DIR/ascend-toolkit/set_env.sh
# source $CANN_DIR/nnal/atb/set_env.sh

# HCCL_NPU_SOCKET_PORT_RANGE / HCCL_CONNECT_TIMEOUT are already exported
# above with cluster-specific values; do not override them here.

# Static torchrun: every node must pass its own --node_rank (resolved
# above), and the master creates the store on MASTER_ADDR:MASTER_PORT.
# Start the master first; workers join immediately after.
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPUS_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG_FILE" 2>&1 | tee "$LOGFILE"
