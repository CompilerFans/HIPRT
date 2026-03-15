# Staged Upstream Sync Patches

These stage patches are grouped so the current MACA fork can be replayed on top of the latest official HIPRT `upstream/main`.

## Base and Target

- Historical fork base: `4baa98e` `Merge pull request #49 from GPUOpen-LibrariesAndSDKs/next-release-10`
- Current fork head for staged export: `8f267ea` `test: add batch geometry diagnostics`
- Intended replay target: `upstream/main` at the time this patch stack was generated

## Stage List

1. `0001-cuda-only.patch`
   - Range: `4baa98e..6caaf4a`
   - Intent: remove HIP-era paths, keep public API surface, converge to CUDA-only build/runtime

2. `0002-maca-core.patch`
   - Range: `6caaf4a..ad1bb28`
   - Intent: first MACA compile/runtime adaptation layer, wave64 fixes, runtime kernel cache controls

3. `0003-maca-functional-pass.patch`
   - Range: `ad1bb28..59e2b74`
   - Intent: scene/traversal/test stabilization until the non-performance unit-test suite reached full pass

4. `0004-logic-restore-and-build-speed.patch`
   - Range: `59e2b74..c550ffe`
   - Intent: restore validated maca_init-aligned scene logic, restore transform-ray baseline, and add build/test acceleration defaults

5. `0005-batch-geometry-diagnostics.patch`
   - Range: `c550ffe..8f267ea`
   - Intent: keep the remaining geometry batch issue visible with dedicated red-light diagnostics

## Suggested Replay Order

1. Apply `0001-cuda-only.patch`
2. Apply `0002-maca-core.patch`
3. Apply `0003-maca-functional-pass.patch`
4. Apply `0004-logic-restore-and-build-speed.patch`
5. Optionally apply `0005-batch-geometry-diagnostics.patch`

## Current Known Status

- Scene-side divergence from `maca_init` has largely been removed again and revalidated.
- The main remaining intentional divergence is the geometry single-object batch fallback.
- The diagnostic conclusion behind `0005` is:
  - unpaired batch geometry fails at `triangleCount=32`
  - pre-paired indexed batch geometry passes at `triangleCount=32`
