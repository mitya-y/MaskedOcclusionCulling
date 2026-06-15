#!/usr/bin/env python3
"""
Batch-run AccuracyBench: fixed list of (model, camera, clip), × dop k.

Default: --headless --max-frames=30; last ACCBench line = stabilized metrics.

CSV: scene_name, bound, camera, prev_frame_depth_mode, all, frustum, mocVis, gpuIds, mocBuf, mocQry, gpuDraw, depthRead.
bound: dop_k=0 -> aabb; dop_k=1 -> aabb-tri; else dop<K> (maps to moc flags as in C++). scene_name = "<scene>_<camera_hash8>".
Failed runs not written; errors only on stderr.

  ./batch_accbench.py --bench ./build/Release/AccuracyBench \\
    --assets-base ../assets --out accbench_batch_results.csv

  # Full frustum GPU id readback (gpuIds without MOC-culled submit):
  ./batch_accbench.py --gpu-draw-always --out accbench_gpu_draw_always.csv

  # Only selected bounds (CSV column `bound`: aabb, aabb-tri, dop10, …):
  ./batch_accbench.py --bounds-filter=aabb,aabb-tri --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Use [^\s]+ for *ms values: prints may be float, nan, inf (old [\d.]+ failed on NaN)
ACCBENCH_RE = re.compile(
    r"ACCBench\s+1\)all=(?P<all>\d+)\s+2\)frustum=(?P<frustum>\d+)\s+3\)mocVis=(?P<mocVis>\d+)\s+4\)gpuIds=(?P<gpuIds>\d+)\s+"
    r"5\)mocBuf=(?P<mocBuf>[^\s]+)ms\s+6\)mocQry=(?P<mocQry>[^\s]+)ms\s+7\)gpuDraw=(?P<gpuDraw>[^\s]+)ms\s+8\)readPx=(?P<readPx>[^\s]+)ms\s+9\)passWall=(?P<passWall>[^\s]+)ms\s+10\)depthRead=(?P<depthRead>[^\s]+)ms\s+11\)guiFrame=(?P<guiFrame>[^\s]+)ms"
)


@dataclass(frozen=True)
class Scene:
    """model_file: basename under --assets-base, or absolute path to .glb / .usd / …"""

    scene_id: str
    model_file: str
    near: float
    far: float
    camera: str


def resolve_model_path(scene: Scene, assets_base: Path) -> Path:
    p = Path(scene.model_file)
    return p if p.is_absolute() else (assets_base / scene.model_file).resolve()


HOTEL_USD = "/home/mitya/Proga/model-renderer/caldera/map_source/prefabs/br/wz_vg/mp_wz_island/commercial/hotel_01.usd"

# testing_cameras snapshot: hotel (USD path) + vikings (.glb under --assets-base)
BENCH_SCENES: list[Scene] = [
    Scene(
        "hotel",
        HOTEL_USD,
        0.5,
        15000.0,
        "((-4995.17, 2422.32, -2190.91), (0.802007, -0.391858, 0.450813), (0.341591, 0.920026, 0.19201))",
    ),
    Scene(
        "hotel",
        HOTEL_USD,
        0.5,
        15000.0,
        "((-1161.94, 529.185, -58.763), (0.872074, -0.0609706, 0.485561), (0.0532756, 0.998139, 0.0296647))",
    ),
    Scene(
        "vikings",
        "vikings.glb",
        0.1,
        1000.0,
        "((157.257, 43.758, 184.509), (0.997816, -0.0596553, -0.0283687), (0.0596308, 0.998219, -0.00169531))",
    ),
    Scene(
        "vikings",
        "vikings.glb",
        0.1,
        1000.0,
        "((511.017, 510.282, 498.346), (-0.371036, -0.865239, -0.337187), (-0.640327, 0.501359, -0.581912))",
    ),
    Scene(
        "vikings",
        "vikings.glb",
        0.1,
        1000.0,
        "((94.5323, 46.3711, 526.099), (0.922742, -0.135941, -0.360648), (0.126615, 0.990717, -0.0494857))",
    ),
    Scene(
        "vikings",
        "vikings.glb",
        0.1,
        1000.0,
        "((91.328, 51.2, 530.305), (0.928073, -0.0830066, -0.363029), (0.077303, 0.996549, -0.0302381))",
    ),
]


def last_accbench_match(text: str) -> re.Match | None:
    last: re.Match | None = None
    for m in ACCBENCH_RE.finditer(text):
        last = m
    return last


def camera_hash8(scene_id: str, camera: str) -> str:
    return hashlib.sha1(f"{scene_id}|{camera}".encode("utf-8")).hexdigest()[:8]


def dop_k_to_moc_bound(dop_k: int) -> tuple[str, str]:
    if dop_k == 0:
        return "aabb", "aabb"
    if dop_k == 1:
        return "aabb-tri", "aabb-tri"
    return f"dop{dop_k}tri", f"dop{dop_k}"


def normalize_bound_token(token: str) -> str:
    """Map filter token to CSV `bound` label (dop10tri -> dop10)."""
    t = token.strip().lower()
    m = re.fullmatch(r"dop(\d+)tri", t)
    if m:
        return f"dop{m.group(1)}"
    return t


def parse_bounds_filter(spec: str | None) -> set[str] | None:
    if spec is None:
        return None
    tokens = [normalize_bound_token(p) for p in spec.split(",") if p.strip()]
    if not tokens:
        return None
    return set(tokens)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", type=Path, default=Path("./build/Release/AccuracyBench"), help="AccuracyBench binary")
    p.add_argument(
        "--assets-base",
        type=Path,
        default=Path("../assets"),
        help="Base dir for relative model_file in BENCH_SCENES (ignored for absolute paths)",
    )
    p.add_argument("--out", type=Path, default=Path("accbench_batch_results.csv"))
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Pass --headless (default: on)")
    p.add_argument(
        "--gpu-draw-always",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --gpu-draw-always: GPU reference draws full frustum (gpuIds without MOC-culled submit)",
    )
    p.add_argument("--max-frames", type=int, default=30, help="Exit after N benchmark frames; last ACCBench used")
    p.add_argument("--dop-min", type=int, default=10)
    p.add_argument("--dop-max", type=int, default=30)
    p.add_argument("--dop-step", type=int, default=2, help="Step from min to max (even k: use step 2)")
    p.add_argument(
        "--bounds-filter",
        metavar="LIST",
        default=None,
        help='Comma-separated bound labels to run (CSV `bound`: aabb, aabb-tri, dop10, …; dop10tri accepted)',
    )
    p.add_argument("--timeout", type=float, default=None, help="Per-run subprocess timeout (seconds)")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args()

    dop_ks: list[int] = [0, 1]
    k = args.dop_min
    while k <= args.dop_max:
        if k not in (0, 1):
            dop_ks.append(k)
        k += args.dop_step

    bounds_filter = parse_bounds_filter(args.bounds_filter)
    if bounds_filter is not None:
        available = {dop_k_to_moc_bound(dop_k)[1] for dop_k in dop_ks}
        unknown = sorted(bounds_filter - available)
        if unknown:
            avail = ", ".join(sorted(available))
            unk = ", ".join(unknown)
            print(
                f"batch_accbench: --bounds-filter unknown bound(s): {unk} (available with current dop range: {avail})",
                file=sys.stderr,
            )
            return 2
        dop_ks = [dop_k for dop_k in dop_ks if dop_k_to_moc_bound(dop_k)[1] in bounds_filter]
        if not dop_ks:
            print("batch_accbench: --bounds-filter matched no bounds in dop range", file=sys.stderr)
            return 2

    fieldnames = [
        "scene_name",
        "bound",
        "camera",
        "prev_frame_depth_mode",
        "all",
        "frustum",
        "mocVis",
        "gpuIds",
        "mocBuf",
        "mocQry",
        "gpuDraw",
        "depthRead",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0

    with args.out.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()

        for camera_index, s in enumerate(BENCH_SCENES):
            model = resolve_model_path(s, args.assets_base)
            scene_name = f"{s.scene_id}_{camera_hash8(s.scene_id, s.camera)}"
            for use_prev_frame_depth in (False, True):
                mode_label = "without-prev-frame-depth" if not use_prev_frame_depth else "with-prev-frame-depth"
                for dop_k in dop_ks:
                    moc, bound = dop_k_to_moc_bound(dop_k)
                    argv: list[str] = [str(args.bench)]
                    if args.headless:
                        argv.append("--headless")
                    if use_prev_frame_depth:
                        argv.append("--use-prev-frame-depth")
                    if args.gpu_draw_always:
                        argv.append("--gpu-draw-always")
                    argv.extend(
                        [
                            str(model),
                            f"--camera={s.camera}",
                            f"--near={s.near}",
                            f"--far={s.far}",
                            f"--moc-test={moc}",
                            f"--max-frames={args.max_frames}",
                        ]
                    )

                    if args.dry_run:
                        print(" ".join(shlex.quote(x) for x in argv))
                        continue

                    try:
                        proc = subprocess.run(
                            argv,
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        n_fail += 1
                        print(
                            f"TIMEOUT {s.scene_id} cam{camera_index} {moc} {mode_label}",
                            file=sys.stderr,
                        )
                        continue

                    err = proc.stderr or ""
                    out = proc.stdout or ""
                    combined = err + "\n" + out
                    m = last_accbench_match(combined)
                    if proc.returncode != 0 or not m:
                        n_fail += 1
                        note = " no_ACCBench_stderr_rebuild?" if proc.returncode == 0 and not m else ""
                        print(
                            f"FAIL rc={proc.returncode} {s.scene_id} cam{camera_index} {moc} {mode_label}{note}",
                            file=sys.stderr,
                        )
                        continue

                    g = m.groupdict()
                    w.writerow(
                        {
                            "scene_name": scene_name,
                            "bound": bound,
                            "camera": s.camera,
                            "prev_frame_depth_mode": mode_label,
                            "all": g["all"],
                            "frustum": g["frustum"],
                            "mocVis": g["mocVis"],
                            "gpuIds": g["gpuIds"],
                            "mocBuf": g["mocBuf"],
                            "mocQry": g["mocQry"],
                            "gpuDraw": g["gpuDraw"],
                            "depthRead": g["depthRead"],
                        }
                    )
                    fcsv.flush()
                    n_ok += 1

    if not args.dry_run:
        print(f"Wrote {args.out}  ok={n_ok}  fail={n_fail}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
