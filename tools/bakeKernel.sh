if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN=python3
fi

mkdir -p hiprt/cache

echo "// automatically generated, don't edit" > hiprt/cache/Kernels.h
echo "// automatically generated, don't edit" > hiprt/cache/KernelArgs.h

echo "#pragma once" >> hiprt/cache/Kernels.h
echo "#pragma once" >> hiprt/cache/KernelArgs.h

$PYTHON_BIN tools/stringify.py ./hiprt/hiprt_vec.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/hiprt_math.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/Obb.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/Aabb.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/AabbList.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/BvhCommon.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/BvhNode.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/Header.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/QrDecomposition.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/Quaternion.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/Transform.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/Instance.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/InstanceList.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/MortonCode.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/TriangleMesh.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/Triangle.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/BvhBuilderUtil.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/SbvhCommon.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/NodeList.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/BvhConfig.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/MemoryArena.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/hiprt_types.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/hiprt_common.h >> hiprt/cache/Kernels.h


# hiprt_device_impl.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/hiprt_device_impl.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/genArgs.py ./hiprt/impl/hiprt_device_impl.h 20220318  >> hiprt/cache/KernelArgs.h

# hiprt_device.h
$PYTHON_BIN tools/stringify.py ./hiprt/hiprt_device.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/genArgs.py ./hiprt/hiprt_device.h 20220318  >> hiprt/cache/KernelArgs.h

# BvhBuilderKernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/BvhBuilderKernels.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/genArgs.py ./hiprt/impl/BvhBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# LbvhBuilderKernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/LbvhBuilderKernels.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/genArgs.py ./hiprt/impl/LbvhBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# PlocBuilderKernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/PlocBuilderKernels.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/genArgs.py ./hiprt/impl/PlocBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# SbvhBuilderKernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/SbvhBuilderKernels.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/genArgs.py ./hiprt/impl/SbvhBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# BatchBuilderKernels.h
$PYTHON_BIN tools/stringify.py ./hiprt/impl/BatchBuilderKernels.h >> hiprt/cache/Kernels.h
$PYTHON_BIN tools/genArgs.py ./hiprt/impl/BatchBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h
