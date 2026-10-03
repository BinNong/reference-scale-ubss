#!/bin/bash
# 远程服务器 rsync/scp 封装（自动从 remote_server.md 读取连接信息与密码）
# 用法:
#   上传: ./scripts/rsync.sh push  <本地路径> <远端路径>
#   下载: ./scripts/rsync.sh pull  <远端路径> <本地路径>
#
# 兜底：本机 OpenSSH 不可用时（`No user exists for uid N` —— 本地 uid 查不到，
# OpenSSH 对 getpwuid 失败直接 fatal，与网络无关）改用 paramiko 通道
# scripts/rssh_py.py。注意 paramiko 版是**逐文件传输**：不做增量、不排除文件，
# 上传目录时也不删除远端多余文件；对本项目足够。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="$ROOT/remote_server.md"

SSH_CMD=$(awk '/^ssh /{print; exit}' "$CONF")
PORT=$(echo "$SSH_CMD" | awk '{for(i=1;i<=NF;i++) if($i=="-p") print $(i+1)}')
TARGET=$(echo "$SSH_CMD" | awk '{print $NF}')
PASS=$(awk -F':[[:space:]]*' '/^password:/{print $2; exit}' "$CONF")

MODE="${1:?用法: rsync.sh push|pull <src> <dst>}"
SRC="$2"
DST="$3"

PYBIN="${RSSH_PYTHON:-$HOME/.workbuddy/binaries/python/envs/default/bin/python}"

# 本机 ssh 可用性探测（一次极轻量往返）
export SSHPASS="$PASS"
probe=$(sshpass -e ssh -p "$PORT" \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR -o ConnectTimeout=10 \
  "$TARGET" 'echo __OK__' 2>&1) || true
case "$probe" in
  *__OK__*) SSH_USABLE=1 ;;
  *)        SSH_USABLE=0 ;;
esac

if [ "$SSH_USABLE" = "1" ]; then
  RSH="sshpass -e ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  case "$MODE" in
    push) exec rsync -az -e "$RSH" "$SRC" "$TARGET:$DST" ;;
    pull) exec rsync -az -e "$RSH" "$TARGET:$SRC" "$DST" ;;
    *)    echo "未知模式: $MODE（应为 push 或 pull）" >&2; exit 1 ;;
  esac
fi

if [ ! -x "$PYBIN" ] || ! "$PYBIN" -c 'import paramiko' 2>/dev/null; then
  echo "本机 ssh 客户端不可用（$probe），且 $PYBIN 里没有 paramiko" >&2
  echo "请先: $PYBIN -m pip install --trusted-host mirrors.aliyun.com paramiko" >&2
  exit 1
fi

case "$MODE" in
  push) exec "$PYBIN" "$ROOT/scripts/rssh_py.py" push "$SRC" "$DST" ;;
  pull) exec "$PYBIN" "$ROOT/scripts/rssh_py.py" pull "$SRC" "$DST" ;;
  *)    echo "未知模式: $MODE（应为 push 或 pull）" >&2; exit 1 ;;
esac
