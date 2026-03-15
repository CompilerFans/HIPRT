//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////////////////

#define HIPRT_BITCODE_LINKING

#include <cuda_runtime.h>
#include <cmath>

#include <hiprt/hiprt_device.h>
#include <test/shared.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

#ifndef SHARED_STACK_SIZE
#define SHARED_STACK_SIZE 1
#endif

extern "C" __global__ void
TraceKernel( hiprtScene scene, uint32_t numOfRays, hiprtGlobalStackBuffer globalStackBuffer, hiprtRay* rays, hiprtHit* hits )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= numOfRays ) return;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, rays[index], stack, instanceStack );
	hits[index] = tr.getNextHit();
}

extern "C" __global__ void CutoutKernel( hiprtGeometry geom, uint8_t* image, hiprtFuncTable table, uint2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	hiprtRay	 ray;
	const float3 o = { x / static_cast<float>( resolution.x ), y / static_cast<float>( resolution.y ), -1.0f };
	const float3 d = { 0.0f, 0.0f, 1.0f };
	ray.origin	   = o;
	ray.direction  = d;

	hiprtGeomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit				  hit = tr.getNextHit();

	image[index * 4 + 0] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 1] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 2] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 3] = 255;
}
