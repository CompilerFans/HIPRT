#define HIPRT_BITCODE_LINKING
#define HIPRT_EXPORTS

#include <cuda_runtime.h>
#include <hiprt/hiprt_device.h>

__device__ bool cutoutFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	(void)ray;
	(void)data;
	(void)payload;
	const float u = hit.uv.x;
	const float v = hit.uv.y;
	const unsigned int checker = static_cast<unsigned int>( u * 16.0f ) + static_cast<unsigned int>( v * 16.0f );
	return ( checker & 1u ) != 0u;
}

extern "C" __global__ void
TraceKernel( hiprtScene scene, uint32_t numOfRays, hiprtGlobalStackBuffer globalStackBuffer, hiprtRay* rays, hiprtHit* hits )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= numOfRays ) return;

	__shared__ uint32_t sharedStackCache[64];
	hiprtSharedStackBuffer sharedStackBuffer{ 1u, sharedStackCache };
	hiprtGlobalStack stack( globalStackBuffer, sharedStackBuffer );
	hiprtEmptyInstanceStack instanceStack;

	hiprtSceneTraversalClosestCustomStack<hiprtGlobalStack, hiprtEmptyInstanceStack> tr( scene, rays[index], stack, instanceStack );
	hits[index] = tr.getNextHit();
}

extern "C" __global__ void CutoutKernel( hiprtGeometry geom, uint8_t* image, hiprtFuncTable table, uint2 resolution )
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= resolution.x || y >= resolution.y ) return;

	const uint32_t index = x + y * resolution.x;
	hiprtRay ray;
	ray.origin = { static_cast<float>( x ) / static_cast<float>( resolution.x ), static_cast<float>( y ) / static_cast<float>( resolution.y ), -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT = 1000.0f;

	hiprtGeomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, nullptr, table );
	const hiprtHit hit = tr.getNextHit();
	image[index * 4 + 0] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 1] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 2] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 3] = 255;
}
