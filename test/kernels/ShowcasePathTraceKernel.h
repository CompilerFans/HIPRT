//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <test/shared.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

#ifndef SHARED_STACK_SIZE
#define SHARED_STACK_SIZE 1
#endif

HIPRT_DEVICE HIPRT_INLINE float luminance( const float3& c ) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

HIPRT_DEVICE HIPRT_INLINE float3 hadamard( const float3& a, const float3& b )
{
	return { a.x * b.x, a.y * b.y, a.z * b.z };
}

HIPRT_DEVICE HIPRT_INLINE float3 clampColor( const float3& c )
{
	return { hiprt::clamp( c.x, 0.0f, 1.0f ), hiprt::clamp( c.y, 0.0f, 1.0f ), hiprt::clamp( c.z, 0.0f, 1.0f ) };
}

HIPRT_DEVICE HIPRT_INLINE float3 skyColor( const float3& dir )
{
	const float t = hiprt::clamp( 0.5f * ( dir.y + 1.0f ), 0.0f, 1.0f );
	return ( 1.0f - t ) * float3{ 0.012f, 0.016f, 0.022f } + t * float3{ 0.085f, 0.105f, 0.135f };
}

HIPRT_DEVICE HIPRT_INLINE float3 rotateAroundYPath( const float3& v, float angle )
{
	const float s = sinf( angle );
	const float c = cosf( angle );
	return { c * v.x + s * v.z, v.y, -s * v.x + c * v.z };
}

HIPRT_DEVICE HIPRT_INLINE float3 sampleAreaLight(
	const Light& light, const float3& x, float3& lightPoint, float3& lightNormal, float& pdf, const float2& xi )
{
	lightNormal	   = hiprt::cross( light.m_lv1 - light.m_lv0, light.m_lv2 - light.m_lv0 );
	const float area = sqrtf( hiprt::dot( lightNormal, lightNormal ) ) * 0.5f;
	lightNormal	   = hiprt::normalize( lightNormal );

	const float2 bary = make_float2( 1.0f - sqrtf( xi.x ), xi.y * sqrtf( xi.x ) );
	lightPoint		   = light.m_lv0 + bary.x * ( light.m_lv1 - light.m_lv0 ) + bary.y * ( light.m_lv2 - light.m_lv0 );

	const float3 r = lightPoint - x;
	const float	dist2 = hiprt::dot( r, r );
	const float cosOnLight = fabs( hiprt::dot( lightNormal, -hiprt::normalize( r ) ) );
	if ( dist2 < 1.0e-6f || cosOnLight < 1.0e-6f || hiprt::dot( r, lightNormal ) > 0.0f )
	{
		pdf = 0.0f;
		return hiprt::make_float3( 0.0f );
	}

	pdf = dist2 * ( 1.0f / area ) / cosOnLight;
	return light.m_le;
}

extern "C" __global__ void __launch_bounds__( 64 ) ShowcasePathTraceKernel(
	hiprtScene			   scene,
	uint8_t*			   image,
	uint2				   resolution,
	hiprtGlobalStackBuffer globalStackBuffer,
	Camera				   camera,
	uint32_t*			   matIndices,
	Material*			   materials,
	uint32_t*			   matOffsetPerInstance,
	uint32_t*			   indices,
	uint32_t*			   indxOffsets,
	float3*				   normals,
	uint32_t*			   normOffset,
	uint32_t*			   numOfLights,
	Light*				   lights,
	float				   lightAngle )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= resolution.x || y >= resolution.y ) return;
	const uint32_t index = x + y * resolution.x;

	constexpr uint32_t Spp		 = 96;
	constexpr uint32_t MaxBounces = 4;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	float3 accumColor = hiprt::make_float3( 0.0f );

	for ( uint32_t sampleIdx = 0; sampleIdx < Spp; ++sampleIdx )
	{
		uint32_t seed = tea<16>( x + y * resolution.x, sampleIdx ).x;
		hiprtRay ray	= generateRay( x, y, resolution, camera, seed, true );

		float3 throughput = hiprt::make_float3( 1.0f );
		float3 radiance   = hiprt::make_float3( 0.0f );

		for ( uint32_t bounce = 0; bounce < MaxBounces; ++bounce )
		{
			hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, ray, stack, instanceStack );
			const hiprtHit hit = tr.getNextHit();

			if ( !hit.hasHit() )
			{
				radiance += hadamard( throughput, skyColor( rotateAroundYPath( ray.direction, lightAngle * 0.35f ) ) );
				break;
			}

			const uint32_t idxOffset = indxOffsets[hit.instanceID];
			const uint32_t idx0		 = indices[idxOffset + ( ( hit.primID * 3 ) + 0 )];
			const uint32_t idx1		 = indices[idxOffset + ( ( hit.primID * 3 ) + 1 )];
			const uint32_t idx2		 = indices[idxOffset + ( ( hit.primID * 3 ) + 2 )];

			const uint32_t normalOffset = normOffset[hit.instanceID];
			const float3   n0			= normals[normalOffset + idx0];
			const float3   n1			= normals[normalOffset + idx1];
			const float3   n2			= normals[normalOffset + idx2];

			float3 Ns = ( 1.0f - hit.uv.x - hit.uv.y ) * n0 + hit.uv.x * n1 + hit.uv.y * n2;
			float3 Ng = hiprtVectorObjectToWorld( hit.normal, scene, hit.instanceID );
			if ( hiprt::dot( ray.direction, Ng ) > 0.0f ) Ng = -Ng;
			Ng = hiprt::normalize( Ng );

			if ( hiprt::dot( Ng, Ns ) < 0.0f ) Ns = Ns - 2.0f * hiprt::dot( Ng, Ns ) * Ng;
			Ns = hiprt::normalize( Ns );

			const uint32_t matOffset	= matOffsetPerInstance[hit.instanceID] + hit.primID;
			const uint32_t matIndex		= matIndices[matOffset];
			const float3   diffuseColor = matIndex == hiprtInvalidValue ? hiprt::make_float3( 0.85f ) : materials[matIndex].m_diffuse;
			const float3   emission		= matIndex == hiprtInvalidValue ? hiprt::make_float3( 0.0f ) : materials[matIndex].m_emission;
			const bool	   isLight		= matIndex != hiprtInvalidValue && materials[matIndex].light();

			if ( isLight )
			{
				radiance += hadamard( throughput, emission * 5.0f );
				break;
			}

			const float3 surfacePt = ray.origin + hit.t * ray.direction;

			if ( numOfLights[0] > 0 )
			{
				const uint32_t lightIndex = min( static_cast<uint32_t>( randf( seed ) * numOfLights[0] ), numOfLights[0] - 1 );
				float3		   lightPoint;
				float3		   lightNormal;
				float		   pdf = 0.0f;
				const float3   le  = sampleAreaLight(
					 lights[lightIndex], surfacePt, lightPoint, lightNormal, pdf, make_float2( randf( seed ), randf( seed ) ) );

				if ( pdf > 0.0f )
				{
					hiprtRay shadowRay;
					shadowRay.origin	= surfacePt + 2.0e-3f * Ng;
					shadowRay.direction = hiprt::normalize( lightPoint - surfacePt );
					shadowRay.maxT		= 0.999f * sqrtf( hiprt::dot( lightPoint - surfacePt, lightPoint - surfacePt ) );

					hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> shadowTraversal(
						scene, shadowRay, stack, instanceStack );
					const bool occluded = shadowTraversal.getNextHit().hasHit();
					if ( !occluded )
					{
						const float nDotL = max( 0.0f, hiprt::dot( Ns, shadowRay.direction ) );
						radiance += hadamard( throughput, diffuseColor * ( nDotL / hiprt::Pi ) * le / pdf ) *
									static_cast<float>( numOfLights[0] );
					}
				}
			}

			const float	 albedo	 = hiprt::clamp( luminance( diffuseColor ), 0.0f, 0.95f );
			const float	 gloss	 = hiprt::clamp( ( albedo - 0.45f ) * 1.45f, 0.0f, 0.72f );
			const float3 reflectDir = hiprt::normalize( ray.direction - 2.0f * hiprt::dot( ray.direction, Ns ) * Ns );
			const float3 diffuseDir = sampleHemisphereCosine( Ns, seed );
			const float3 nextDir	= hiprt::normalize( ( 1.0f - gloss ) * diffuseDir + gloss * reflectDir );

			const float3 tint = ( 1.0f - gloss ) * diffuseColor + gloss * float3{ 0.92f, 0.90f, 0.88f };
			throughput		 = hadamard( throughput, tint );

			if ( bounce >= 2 )
			{
				const float rr = hiprt::clamp( max( throughput.x, max( throughput.y, throughput.z ) ), 0.1f, 0.95f );
				if ( randf( seed ) > rr ) break;
				throughput = throughput * ( 1.0f / rr );
			}

			ray.origin	 = surfacePt + 2.0e-3f * Ng;
			ray.direction = nextDir;
			ray.maxT		 = 1.0e6f;
		}

		accumColor += radiance;
	}

	float3 finalColor = clampColor( gammaCorrect( accumColor * ( 1.0f / static_cast<float>( Spp ) ) ) );
	image[index * 4 + 0] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.x ), 0u, 255u );
	image[index * 4 + 1] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.y ), 0u, 255u );
	image[index * 4 + 2] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.z ), 0u, 255u );
	image[index * 4 + 3] = 255;
}
