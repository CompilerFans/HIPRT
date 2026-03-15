# Upstream Main Upgrade Status

## Current Setup

- Patch stack directory:
  - `/data/HIPRT/patches/staged-upstream-sync-20260315`
- Official latest base checked:
  - `upstream/main`
  - commit `16d7899` `Merge pull request #62 from jpola-amd/jpola/dev/unroll-O2W`
- Integration work directory prepared:
  - `/data/HIPRT_upstream_apply`
- Integration branch created there:
  - `upstream-main-maca-20260315`

## Stage Patch Apply Check Result

All five staged patches were generated successfully, but a direct `git apply --check` on top of official latest `upstream/main` does **not** apply cleanly.

### 0001-cuda-only.patch

- Status: `apply --check` failed
- Main conflict areas:
  - `CMakeLists.txt`
  - `hiprt/hiprt_types.h`
  - `hiprt/hiprtew.h.in`
  - `hiprt/impl/Compiler.cpp`
  - `hiprt/impl/Context.cpp`
  - `hiprt/impl/Kernel.*`
  - `hiprt/impl/*Builder*.h`
  - `test/hiprtTest.*`
  - `premake5.lua`
  - `scripts/bitcodes/*`
  - `contrib/Orochi/*`
- Interpretation:
  - official latest has drifted too far from the original `3.0.2.4242e39` base for the full CUDA-only convergence patch to apply as a single bulk patch

### 0002-maca-core.patch

- Status: `apply --check` failed
- Main conflict areas:
  - `CMakeLists.txt`
  - `README.md`
  - `hiprt/hiprt_common.h`
  - `hiprt/impl/BvhNode.h`
  - `hiprt/impl/Compiler.cpp`
  - `hiprt/impl/Context.cpp`
- Interpretation:
  - core MACA bring-up overlaps directly with files changed by official latest

### 0003-maca-functional-pass.patch

- Status: `apply --check` failed
- Main conflict areas:
  - `CMakeLists.txt`
  - `README.md`
  - `hiprt/impl/Compiler.*`
  - `hiprt/impl/Context.cpp`
  - `hiprt/impl/Transform.h`
  - `hiprt/impl/hiprt_device_impl.h`
  - `test/hiprtTest.cpp`
  - `test/shared.h`
  - docs paths absent on upstream latest
- Interpretation:
  - the functional-fix stage depends on the earlier CUDA-only/MACA core layout and cannot be applied independently on official latest

### 0004-logic-restore-and-build-speed.patch

- Status: `apply --check` failed
- Main conflict areas:
  - `CMakeLists.txt`
  - `README.md`
  - `hiprt/hiprt_common.h`
  - `hiprt/impl/BvhBuilderKernels.h`
  - `hiprt/impl/BvhNode.h`
  - `hiprt/impl/Context.cpp`
  - `hiprt/impl/Transform.h`
  - `hiprt/impl/hiprt_device_impl.h`
  - `scripts/build.sh`
  - `scripts/unittest*.sh`
  - docs paths absent on upstream latest
- Interpretation:
  - this stage is logically later and should be replayed only after the upstream-side baseline shape is re-established

### 0005-batch-geometry-diagnostics.patch

- Status: `apply --check` failed
- Main conflict areas:
  - `test/hiprtTest.*`
  - `test/main.cpp`
  - docs paths absent on upstream latest
- Interpretation:
  - diagnostics are the easiest stage conceptually, but even they depend on earlier test harness and doc layout changes

## Practical Conclusion

The patch stack is useful as a **replay plan and stage reference**, but not as a direct one-shot apply on official latest `upstream/main`.

The upgrade should proceed as a **manual staged replay**:

1. Rebuild the CUDA-only convergence on top of official latest, file by file.
2. Re-introduce the MACA core bring-up on the new upstream layout.
3. Re-apply only the still-needed functional fixes.
4. Re-skip the already validated “unnecessary modifications” while replaying.
5. Re-add the batch geometry diagnostics at the end.

## Recommended Next Merge Order

1. Replay the CUDA-only convergence first:
   - build system
   - public API preservation
   - Orochi trimming / HIP path removal
2. Replay the minimum MACA compile/runtime layer:
   - `hiprt_common`
   - compiler/runtime control
   - wave64 handling
3. Re-run unit tests on the new upstream base before replaying later fixes.
4. Replay only the still-needed scene/transform fixes selectively:
   - current analysis already showed many of the old detours are no longer needed
5. Re-add the batch geometry diagnostics last and continue root-cause work there.
