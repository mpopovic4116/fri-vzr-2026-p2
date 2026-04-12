# pyright: strict


from collections import defaultdict
from collections.abc import Mapping
from sys import stdin


def _parse_kv_single(kv: str) -> tuple[str, str] | None:
    xs = kv.split("=", 1)
    if len(xs) != 2:
        return None
    return (xs[0], xs[1])


def parse_kv_line(line: str) -> dict[str, str]:
    return {
        kv[0]: kv[1]
        for kv in (_parse_kv_single(kv) for kv in line.split())
        if kv is not None
    }


def main_kvmean():
    current_key: frozenset[tuple[str, str]] = frozenset()
    current_key_order: list[str] = []
    current_vals: defaultdict[str, float] = defaultdict(float)
    current_n = 0

    def flush(key_mut: Mapping[str, str] | None = None):
        nonlocal current_key
        nonlocal current_key_order
        nonlocal current_vals
        nonlocal current_n

        if key_mut is not None:
            key = frozenset(key_mut.items())
            if key == current_key:
                return False

        if current_n > 0:
            out: list[str] = []
            current_key_mut = dict(current_key)
            for k in current_key_order:
                out.append(f"{k}={current_key_mut[k]}")
            for k, v in current_vals.items():
                out.append(f"{k}={v / current_n:0.15f}")
            print(" ".join(out), flush=True)

        if key_mut is None:
            current_key = frozenset()
            current_key_order = []
        else:
            current_key = key  # pyright: ignore[reportPossiblyUnboundVariable]
            current_key_order = list(key_mut.keys())
        current_vals = defaultdict(float)
        current_n = 0

        return True

    for line in stdin:
        line_key_mut: dict[str, str] = {}
        line_vals: dict[str, float] = {}
        parsed = parse_kv_line(line)
        for k, v in parsed.items():
            if k.startswith("t_"):
                line_vals[k] = float(v)
            elif k != "i":
                line_key_mut[k] = v
        if flush(line_key_mut):
            for k, v in line_vals.items():
                current_vals[k] += v
            current_n += 1
    flush()
