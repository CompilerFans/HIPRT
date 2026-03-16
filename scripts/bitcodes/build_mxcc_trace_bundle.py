#!/usr/bin/env python3

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile


def run(cmd: list[str], cwd: pathlib.Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def default_compiler() -> str:
    return "/opt/maca/mxgpu_llvm/bin/mxcc"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a loadable MACA bundle for user trace kernels with mxcc --maca-link."
    )
    parser.add_argument("--root", required=True, help="HIPRT repository root")
    parser.add_argument("--source", required=True, help="User kernel source path")
    parser.add_argument("--output", required=True, help="Output fatbin/bundle path")
    parser.add_argument("--compiler", default=default_compiler(), help="mxcc path")
    parser.add_argument("--offload-arch", default="xcore1000", help="MXCC offload arch")
    parser.add_argument(
        "--with-default-func-table",
        action="store_true",
        help="Inject default intersectFunc/filterFunc wrappers for kernels that do not provide custom tables",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    source = pathlib.Path(args.source).resolve()
    output = pathlib.Path(args.output).resolve()
    compiler = pathlib.Path(args.compiler).resolve()

    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")
    if not compiler.exists():
        raise FileNotFoundError(f"compiler not found: {compiler}")

    maca_path = pathlib.Path("/opt/maca")
    cuda_path = pathlib.Path("/root/cu-bridge/CUDA_DIR")
    if not cuda_path.exists():
        alt = maca_path / "tools" / "cu-bridge" / "CUDA_DIR"
        cuda_path = alt if alt.exists() else maca_path / "tools" / "cu-bridge"

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hiprt_mxcc_bundle_", dir=str(output.parent)) as temp_dir_raw:
        temp_dir = pathlib.Path(temp_dir_raw)
        wrapper = temp_dir / "trace_bundle_wrapper.cu"
        obj = temp_dir / "trace_bundle.o"

        wrapper_lines = [
            "#define HIPRT_BITCODE_LINKING",
            "#include <hiprt/hiprt_device.h>",
        ]
        if args.with_default_func_table:
            wrapper_lines.extend(
                [
                    "HIPRT_DEVICE bool intersectFunc(",
                    "    uint32_t geomType,",
                    "    uint32_t rayType,",
                    "    const hiprtFuncTableHeader& tableHeader,",
                    "    const hiprtRay& ray,",
                    "    void* payload,",
                    "    hiprtHit& hit )",
                    "{",
                    "    (void)geomType; (void)rayType; (void)tableHeader; (void)ray; (void)payload; (void)hit;",
                    "    return false;",
                    "}",
                    "HIPRT_DEVICE bool filterFunc(",
                    "    uint32_t geomType,",
                    "    uint32_t rayType,",
                    "    const hiprtFuncTableHeader& tableHeader,",
                    "    const hiprtRay& ray,",
                    "    void* payload,",
                    "    const hiprtHit& hit )",
                    "{",
                    "    (void)geomType; (void)rayType; (void)tableHeader; (void)ray; (void)payload; (void)hit;",
                    "    return false;",
                    "}",
                ]
            )
        wrapper_lines.extend(
            [
                "#include <hiprt/impl/hiprt_device_impl.h>",
                f'#include "{source}"',
            ]
        )
        wrapper.write_text("\n".join(wrapper_lines) + "\n", encoding="utf-8")

        common = [
            str(compiler),
            "-O3",
            "-std=c++17",
            "-x",
            "maca",
            "-fgpu-rdc",
            "--include",
            "cuda_runtime.h",
            "-D__CUDACC__",
            f"-I{root}",
            f"-I{root / 'test'}",
            f"-I{root / 'contrib' / 'Orochi'}",
            f"-I{cuda_path / 'include'}",
            f"-I{maca_path / 'include'}",
            "--offload-arch=" + args.offload_arch,
        ]

        run(common + ["-c", str(wrapper), "-o", str(obj)], root)
        run([str(compiler), "-fgpu-rdc", "--maca-link", str(obj), "-fatbin", "-o", str(output)], root)

    if not output.exists():
        raise RuntimeError(f"missing output: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"build_mxcc_trace_bundle.py failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"build_mxcc_trace_bundle.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
