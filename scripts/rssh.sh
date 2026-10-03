#!/bin/bash
# 远程服务器 SSH 封装：自动从 remote_server.md 读取连接信息与密码
# 用法: ./scripts/rssh.sh '命令'
#
# 兜底：若本机 OpenSSH 客户端不可用（典型症状 `No user exists for uid N` —— 本地 uid 在
# passwd/OpenDirectory 里查不到，而 OpenSSH 对 getpwuid(getuid()) 失败直接 fatal，
# 与网络和服务器无关），则自动改用基于 paramiko 的通道 scripts/rssh_py.py。
# 该通道需要：受管 python 环境里装了 paramiko（pip install --trusted-host mirrors.aliyun.com paramiko）。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="$ROOT/remote_server.md"

SSH_CMD=$(awk '/^ssh /{print; exit}' "$CONF")
PORT=$(echo "$SSH_CMD" | awk '{for(i=1;i<=NF;i++) if($i=="-p") print $(i+1)}')
TARGET=$(echo "$SSH_CMD" | awk '{print $NF}')
PASS=$(awk -F':[[:space:]]*' '/^password:/{print $2; exit}' "$CONF")

if [ -z "$PORT" ] || [ -z "$TARGET" ] || [ -z "$PASS" ]; then
  echo "无法解析 remote_server.md 中的连接信息" >&2
  exit 1
fi

PYBIN="${RSSH_PYTHON:-$HOME/.workbuddy/binaries/python/envs/default/bin/python}"

# 探测本机 ssh 是否可用：只跑一次极轻量的探测，失败即切换通道。
export SSHPASS="$PASS"
probe=$(sshpass -e ssh -p "$PORT" \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR -o ConnectTimeout=10 -o ServerAliveInterval=30 \
  "$TARGET" 'echo __OK__' 2>&1) || true
case "$probe" in
  *__OK__*) SSH_USABLE=1 ;;
  *)        SSH_USABLE=0 ;;
esac

if [ "$SSH_USABLE" = "1" ]; then
  exec sshpass -e ssh -p "$PORT" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o ServerAliveInterval=30 \
    "$TARGET" "$@"
fi

if [ ! -x "$PYBIN" ] || ! "$PYBIN" -c 'import paramiko' 2>/dev/null; then
  echo "本机 ssh 客户端不可用（$probe）" >&2
  echo "且 $PYBIN 里没有 paramiko；请先：" >&2
  echo "  $PYBIN -m pip install --trusted-host mirrors.aliyun.com paramiko" >&2
  exit 1
fi

# 走 paramiko 兜底通道：命令经 argv 传参，避免二次 shell 解析
exec "$PYBIN" "$ROOT/scripts/rssh_py.py" run "$@"
