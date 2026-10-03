#!/usr/bin/env python3
"""基于 paramiko 的远程执行/传输通道，用于**本地 ssh 客户端不可用**时的兜底。

背景：本机 uid 501 在 passwd/OpenDirectory 里查不到（`pwd.getpwuid(501)` → KeyError），
而 OpenSSH 客户端在 `main()` 里对 `getpwuid(getuid())` 失败**直接 fatal**
（`No user exists for uid 501`），与网络和服务器无关。`/usr/bin/ssh` 受 SIP 保护，
无法用 DYLD 注入绕过；sudo 也不可用。paramiko 不查本地 passwd，因此可以绕开。

用法（与 rssh.sh / rsync.sh 对应）：
    python3 scripts/rssh_py.py run '命令'              # 执行远程命令（shell）
    python3 scripts/rssh_py.py run < script.sh         # 从 stdin 读脚本执行
    python3 scripts/rssh_py.py push <本地目录> <远端目录>
    python3 scripts/rssh_py.py pull <远端目录> <本地目录>

连接信息仍从 remote_server.md 读取，格式不变。
"""
from __future__ import annotations

import os
import posixpath
import stat
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "remote_server.md"


def profile() -> tuple[str, int, str, str]:
    user = host = pw = ""
    port = 22
    for line in CONF.read_text().splitlines():
        line = line.strip()
        if line.startswith("ssh "):
            toks = line.split()
            for i, t in enumerate(toks):
                if t == "-p":
                    port = int(toks[i + 1])
            target = toks[-1]
            if "@" in target:
                user, host = target.split("@", 1)
            else:
                host = target
        elif line.lower().startswith("password:"):
            pw = line.split(":", 1)[1].strip()
        elif line.startswith("user:"):
            user = line.split(":", 1)[1].strip()
    if not (user and host and pw):
        sys.exit("无法从 remote_server.md 解析 user/host/password")
    return host, port, user, pw


def connect() -> paramiko.SSHClient:
    host, port, user, pw = profile()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname=host, port=port, username=user, password=pw,
              timeout=20, banner_timeout=20, auth_timeout=20,
              look_for_keys=False, allow_agent=False)
    return c


def run(cmd: str | None = None) -> int:
    if cmd is None:
        cmd = sys.stdin.read()
    c = connect()
    try:
        # 用 get_pty=False + exec_command；stdout/stderr 分流，退出码透传
        _, out, err = c.exec_command(cmd)
        rc = out.channel.recv_exit_status()
        sys.stdout.write(out.read().decode(errors="replace"))
        sys.stdout.flush()
        e = err.read().decode(errors="replace")
        if e.strip():
            sys.stderr.write(e)
        return rc
    finally:
        c.close()


def _sftp_walk(sftp: paramiko.SFTPClient, remote: str):
    """深度优先遍历远端目录，产出 (远端路径, 是否目录)。"""
    for e in sorted(sftp.listdir_attr(remote), key=lambda a: a.filename):
        p = posixpath.join(remote, e.filename)
        if stat.S_ISDIR(e.st_mode):
            yield p, True
            yield from _sftp_walk(sftp, p)
        else:
            yield p, False


def push(local: str, remote: str) -> int:
    lp = Path(local)
    c = connect()
    try:
        sftp = c.open_sftp()

        def put_dir(src: Path, dst: str):
            try:
                sftp.mkdir(dst)
            except OSError:
                pass
            for e in sorted(src.iterdir()):
                tar = posixpath.join(dst, e.name)
                if e.is_dir():
                    put_dir(e, tar)
                elif e.is_file():
                    sftp.put(str(e), tar)

        if lp.is_dir():
            put_dir(lp, remote.rstrip("/"))
            print(f"已上传 {lp} -> {remote}")
        else:
            # 远端是已存在的目录时，落到该目录下的同名文件——与 scp/rsync 一致。
            # 原实现直接 put 到目录路径，静默失败/报错（pull 端同类问题已修过）。
            dst = remote
            try:
                if stat.S_ISDIR(sftp.stat(remote).st_mode):
                    dst = posixpath.join(remote.rstrip("/"), lp.name)
            except OSError:
                pass
            sftp.put(str(lp), dst)
            print(f"已上传 {lp} -> {dst}")
        sftp.close()
        return 0
    finally:
        c.close()


def pull(remote: str, local: str) -> int:
    c = connect()
    try:
        sftp = c.open_sftp()
        st = sftp.stat(remote)
        lf = Path(local)

        if stat.S_ISDIR(st.st_mode):
            n = 0
            for rp, isdir in _sftp_walk(sftp, remote):
                rel = posixpath.relpath(rp, remote)
                dst = lf / rel
                if isdir:
                    dst.mkdir(parents=True, exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(rp, str(dst))
                    n += 1
            print(f"已下载 {n} 个文件 {remote} -> {local}")
        else:
            # 目标是已存在的目录时，落到该目录下的同名文件——与 rsync/scp 的行为一致。
            # 原实现直接 sftp.get 到目录路径，报 IsADirectoryError（实测踩过）。
            if lf.is_dir():
                lf = lf / posixpath.basename(remote.rstrip("/"))
            lf.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, str(lf))
            print(f"已下载 {remote} -> {lf}")
        sftp.close()
        return 0
    finally:
        c.close()


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "run":
        return run(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    if cmd == "push" and len(sys.argv) == 4:
        return push(sys.argv[2], sys.argv[3])
    if cmd == "pull" and len(sys.argv) == 4:
        return pull(sys.argv[2], sys.argv[3])
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
