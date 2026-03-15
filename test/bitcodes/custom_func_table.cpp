//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////////////////

#define HIPRT_BITCODE_LINKING
#define HIPRT_EXPORTS

#include <cuda_runtime.h>
#include <cmath>

#include <hiprt/hiprt_device.h>

__device__ bool duplicityFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit );
__device__ bool intersectCircle( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit );
__device__ bool intersectSphere( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit );
__device__ bool cutoutFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit );

HIPRT_DEVICE HIPRT_INLINE bool intersectSphere( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float3 orig = ray.origin;
	const float3 dir  = ray.direction;

	const float4 sphere = reinterpret_cast<const float4*>( data )[hit.primID];
	const float3 center = hiprt::make_float3( sphere );
	const float	 radius = sphere.w;

	const float3 O = orig - center;
	const float3 D = hiprt::normalize( dir );

	const float b	 = hiprt::dot( O, D );
	const float c	 = hiprt::dot( O, O ) - radius * radius;
	const float disc = b * b - c;
	if ( disc > 0.0f )
	{
		const float sdisc = sqrtf( disc );
		const float root  = -b - sdisc;
		hit.t			  = root;
		hit.normal		  = ( O + root * D ) / radius;
		return true;
	}

	return false;
}

HIPRT_DEVICE HIPRT_INLINE bool intersectCircle( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float*	o = reinterpret_cast<const float*>( data );
	constexpr float R = 0.1f;

	const float2 c		= { o[hit.primID] - ray.origin.x, 0.5f - ray.origin.y };
	const float	 d		= sqrtf( c.x * c.x + c.y * c.y );
	const bool	 hasHit = d < R;

	uint2 colors[] = { { 255, 0 }, { 0, 255 }, { 255, 255 } };
	if ( hasHit && payload )
	{
		uint2* color = reinterpret_cast<uint2*>( payload );
		*color		 = colors[hit.primID];
	}

	return hasHit;
}

HIPRT_DEVICE HIPRT_INLINE bool duplicityFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	uint32_t* processed = reinterpret_cast<uint32_t*>( payload );
	if ( processed[hit.primID] ) return true;
	processed[hit.primID] = 1u;
	return false;
}

HIPRT_DEVICE HIPRT_INLINE bool cutoutFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	const float	  scale = 16.0f;
	const float2& uv	  = hit.uv;
	float2		  texCoord[2];
	texCoord[0] = ( 1.0f - uv.x - uv.y ) * float2{ 0.0f, 0.0f } + uv.x * float2{ 0.0f, 1.0f } + uv.y * float2{ 1.0f, 1.0f };
	texCoord[1] = ( 1.0f - uv.x - uv.y ) * float2{ 0.0f, 0.0f } + uv.x * float2{ 1.0f, 1.0f } + uv.y * float2{ 1.0f, 0.0f };
	return ( static_cast<uint32_t>( scale * texCoord[hit.primID].x ) + static_cast<uint32_t>( scale * texCoord[hit.primID].y ) ) &
		   1;
}

HIPRT_DEVICE bool intersectFunc(
	uint32_t					geomType,
	uint32_t					rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay&				ray,
	void*						payload,
	hiprtHit&					hit )
{
	const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
	const void*	   data	 = tableHeader.funcDataSets[index].intersectFuncData;
	switch ( index )
	{
	case 0: return intersectCircle( ray, data, payload, hit );
	case 1: return intersectSphere( ray, data, payload, hit );
	default: return false;
	}
}

HIPRT_DEVICE bool filterFunc(
	uint32_t					geomType,
	uint32_t					rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay&				ray,
	void*						payload,
	const hiprtHit&				hit )
{
	const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
	const void*	   data	 = tableHeader.funcDataSets[index].filterFuncData;
	switch ( index )
	{
	case 2: return duplicityFilter( ray, data, payload, hit );
	case 3: return cutoutFilter( ray, data, payload, hit );
	default: return false;
	}
}
