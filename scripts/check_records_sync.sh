#!/bin/bash
# 校验 results/ 与服务器端归档是否**逐文件一致**。
#
# 动机：本地归档曾与服务器不同步——validate_guard.py 的决策口径修好后在服务器重跑，
# 但 results/guard_validation.json 没有拉回来，于是本地记录里 Laplace 条件仍是
# decision=OFF / n_wrong=1，而手稿已按修正后的 0/27 改写。这类"记录没跟着重跑"
# 是复现性硬伤：别人按记录复现，得到的手稿数字是错的。
#
# 判据：md5 逐文件比对。用法：
#     ./scripts/check_records_sync.sh          # 比对 servers/results 与本机 results
# 退出码非 0 表示存在差异（或在无法连服务器时返回 2）。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_RESULTS="/data/experiment/paper5_ubss_sadun/results"

"$ROOT/scripts/rssh.sh" "cd $REMOTE_RESULTS && md5sum *.json 2>/dev/null | sort -k2" \
  > /tmp/_srv_hash.txt || { echo "无法读取服务器归档"; exit 2; }
( cd "$ROOT/results" && md5 -r *.json 2>/dev/null | awk '{print $1" "$2}' | sort -k2 ) \
  > /tmp/_loc_hash.txt || { echo "无法读取本地归档"; exit 2; }

/usr/local/bin/python3 - <<'PY'
import sys
def load(p):
    d = {}
    for l in open(p):
        parts = l.split()
        if len(parts) == 2:
            d[parts[1].replace("*", "")] = parts[0]
    return d

srv, loc = load("/tmp/_srv_hash.txt"), load("/tmp/_loc_hash.txt")
diff = [f for f in sorted(set(srv) & set(loc)) if srv[f] != loc[f]]
only_s = sorted(set(srv) - set(loc))
only_l = sorted(set(loc) - set(srv))

print(f"共比对 {len(set(srv) | set(loc))} 个记录")
for f in diff:   print(f"  ✗ 内容不同: {f}")
for f in only_s: print(f"  ✗ 仅服务器有: {f}")
for f in only_l: print(f"  ✗ 仅本地有: {f}")
if not (diff or only_s or only_l):
    print("  ✓ 全部一致")
    sys.exit(0)
print("\n差异意味着手稿可能引用了拿不到的那一版数字。"
      "先用 rsync 把服务器上的记录拉回来，再重跑 verify_revision.py。")
sys.exit(1)
PY
