#!/usr/bin/env python3

import argparse
import pathlib
import shutil
import subprocess
import sys


def read_version(root: pathlib.Path) -> str:
    with (root / "version.txt").open("r", encoding="utf-8") as f:
        major = int(f.readline().strip())
        minor = int(f.readline().strip())
    return f"{major * 1000 + minor:05d}"


def parse_arch_list(arch_list: str) -> list[str]:
    archs = [item.strip() for item in arch_list.replace(",", ";").split(";") if item.strip()]
    if not archs:
        return ["80"]
    return archs


def build_gencode_flags(archs: list[str]) -> list[str]:
    flags: list[str] = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags


def run(cmd: list[str], cwd: pathlib.Path, dst: pathlib.Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)
    if not dst.exists():
        raise RuntimeError(f"missing output: {dst}")


def copy_outputs(outputs: list[pathlib.Path], destinations: list[pathlib.Path]) -> None:
    for dst_dir in destinations:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for output in outputs:
            shutil.copy2(output, dst_dir / output.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HIPRT precompiled fatbin artifacts for CUDA/cu-bridge.")
    parser.add_argument("--root", required=True, help="Repository root")
    parser.add_argument("--compiler", default="nvcc", help="CUDA-compatible compiler, e.g. nvcc or cucc")
    parser.add_argument("--config", default="Release", help="Build config name used for dist/bin/<config>")
    parser.add_argument("--arch-list", default="", help="CUDA arch list such as 75;80;86;89")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    workdir = pathlib.Path(__file__).resolve().parent
    version = read_version(root)
    arch_flags = build_gencode_flags(parse_arch_list(args.arch_list))
    config_dir = root / "dist" / "bin" / args.config
    bitcode_dir = root / "hiprt" / "bitcodes"

    common_flags = [
        args.compiler,
        "-x",
        "cu",
        "-O3",
        "-std=c++17",
        "-fatbin",
        "-I../../contrib/Orochi/",
        "-I../../",
        "-DHIPRT_BITCODE_LINKING",
        "--use_fast_math",
    ] + arch_flags

    outputs: list[pathlib.Path] = []

    hiprt_lib = workdir / f"hiprt{version}_nv_lib.fatbin"
    run(
        common_flags
        + ["--device-c", "../../hiprt/impl/hiprt_kernels_bitcode.h", "-o", str(hiprt_lib)],
        workdir,
        hiprt_lib,
    )
    outputs.append(hiprt_lib)

    hiprt_fatbin = workdir / f"hiprt{version}_nv.fatbin"
    run(
        common_flags
        + ["../../hiprt/impl/hiprt_kernels.h", "-o", str(hiprt_fatbin)],
        workdir,
        hiprt_fatbin,
    )
    outputs.append(hiprt_fatbin)

    copy_outputs(outputs, [config_dir, bitcode_dir])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"compile.py failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"compile.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
