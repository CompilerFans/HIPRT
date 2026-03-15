#!/usr/bin/env python3

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


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


def is_mxcc(compiler: str, toolchain: str) -> bool:
    if toolchain:
        return toolchain == "mxcc"
    return pathlib.Path(compiler).name == "mxcc"


def build_mxcc_flags(root: pathlib.Path) -> list[str]:
    maca_path = pathlib.Path(os.environ.get("MACA_PATH", "/opt/maca"))
    cuda_path = pathlib.Path(os.environ.get("CUDA_PATH", str(maca_path / "tools" / "cu-bridge")))
    offload_arch = os.environ.get("MXCC_OFFLOAD_ARCH", "xcore1000")
    return [
        "-x",
        "maca",
        "-fgpu-rdc",
        "--include",
        "cuda_runtime.h",
        "-D__CUDACC__",
        "-I../../",
        "-I../../contrib/Orochi/",
        f"-I{cuda_path / 'include'}",
        f"-I{maca_path / 'include'}",
        "--offload-arch=" + offload_arch,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a precompiled trace-kernel fatbin for CUDA/cu-bridge.")
    parser.add_argument("--root", required=True, help="Repository root")
    parser.add_argument("--compiler", default="nvcc", help="CUDA-compatible compiler, e.g. nvcc or cucc")
    parser.add_argument("--toolchain", choices=["auto", "nvcc", "mxcc"], default="auto", help="Offline compiler flavor")
    parser.add_argument("--config", default="Release", help="Build config name used for dist/bin/<config>")
    parser.add_argument("--arch-list", default="", help="CUDA arch list such as 75;80;86;89")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    workdir = pathlib.Path(__file__).resolve().parent
    version = read_version(root)
    arch_flags = build_gencode_flags(parse_arch_list(args.arch_list))

    config_dir = root / "dist" / "bin" / args.config
    bitcode_dir = root / "hiprt" / "bitcodes"
    output = workdir / f"hiprt{version}_nv_precompiled_bitcode.fatbin"

    with tempfile.TemporaryDirectory(prefix="hiprt_precompile_", dir=str(workdir)) as temp_dir:
        source = pathlib.Path(temp_dir) / "precompiled_trace_kernel.cu"
        source.write_text(
            "#define HIPRT_EXPORTS\n"
            '#include "../../hiprt/impl/hiprt_kernels_bitcode.h"\n'
            '#include "../../test/bitcodes/custom_func_table.cpp"\n'
            '#include "../../test/bitcodes/unit_test.cpp"\n',
            encoding="utf-8",
        )

        use_mxcc = is_mxcc(args.compiler, "" if args.toolchain == "auto" else args.toolchain)
        if use_mxcc:
            cmd = [
                args.compiler,
                "-O3",
                "-std=c++17",
                "-fatbin",
                "-DHIPRT_BITCODE_LINKING",
                "-use-fast-math",
            ] + build_mxcc_flags(root) + [str(source), "-o", str(output)]
        else:
            cmd = [
                args.compiler,
                "-x",
                "cu",
                str(source),
                "-O3",
                "-std=c++17",
                "-fatbin",
                "-I../../",
                "-I../../contrib/Orochi/",
                "-DHIPRT_BITCODE_LINKING",
                "--use_fast_math",
            ] + arch_flags + ["-o", str(output)]

        print(" ".join(cmd))
        subprocess.run(cmd, cwd=workdir, check=True)

    if not output.exists():
        raise RuntimeError(f"missing output: {output}")

    for dst_dir in (config_dir, bitcode_dir):
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, dst_dir / output.name)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"precompile_bitcode.py failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"precompile_bitcode.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
