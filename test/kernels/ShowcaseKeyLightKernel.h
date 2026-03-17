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

HIPRT_DEVICE HIPRT_INLINE float3 rotateAroundY( const float3& v, float angle )
{
	const float s = sinf( angle );
	const float c = cosf( angle );
	return { c * v.x + s * v.z, v.y, -s * v.x + c * v.z };
}

extern "C" __global__ void __launch_bounds__( 64 ) ShowcaseKeyLightKernel(
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
	const uint32_t index = x + y * resolution.x;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	uint32_t seed = tea<16>( x + y * resolution.x, 7 ).x;
	hiprtRay ray	= generateRay( x, y, resolution, camera, seed, false );
	hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, ray, stack, instanceStack );
	hiprtHit hit = tr.getNextHit();

	float3 finalColor{};
	if ( hit.hasHit() )
	{
		const uint32_t idxOffset = indxOffsets[hit.instanceID];
		const uint32_t idx0		 = indices[idxOffset + ( ( hit.primID * 3 ) + 0 )];
		const uint32_t idx1		 = indices[idxOffset + ( ( hit.primID * 3 ) + 1 )];
		const uint32_t idx2		 = indices[idxOffset + ( ( hit.primID * 3 ) + 2 )];

		const uint32_t nOffset = normOffset[hit.instanceID];
		const float3   n0	   = normals[nOffset + idx0];
		const float3   n1	   = normals[nOffset + idx1];
		const float3   n2	   = normals[nOffset + idx2];

		float3 Ns = ( 1.0f - hit.uv.x - hit.uv.y ) * n0 + hit.uv.x * n1 + hit.uv.y * n2;
		float3 Ng = hiprtVectorObjectToWorld( hit.normal, scene, hit.instanceID );
		if ( hiprt::dot( ray.direction, Ng ) > 0.0f ) Ng = -Ng;
		Ng = hiprt::normalize( Ng );

		if ( hiprt::dot( Ng, Ns ) < 0.0f ) Ns = Ns - 2.0f * hiprt::dot( Ng, Ns ) * Ng;
		Ns = hiprt::normalize( Ns );

		const uint32_t matOffset	= matOffsetPerInstance[hit.instanceID] + hit.primID;
		const uint32_t matIndex		= matIndices[matOffset];
		const float3   diffuseColor	= matIndex == hiprtInvalidValue ? hiprt::make_float3( 1.0f ) : materials[matIndex].m_diffuse;
		const float3   emissiveColor = matIndex == hiprtInvalidValue ? hiprt::make_float3( 0.0f ) : materials[matIndex].m_emission;

		if ( matIndex != hiprtInvalidValue && materials[matIndex].light() )
		{
			finalColor = emissiveColor * 3.5f;
		}
		else
		{
			const float3 surfacePt = ray.origin + hit.t * ray.direction;
			const float3 toLight   = hiprt::normalize( rotateAroundY( float3{ -0.55f, 0.85f, 0.35f }, lightAngle ) );
			const float3 fillLight = hiprt::normalize( rotateAroundY( float3{ 0.45f, 0.25f, 0.85f }, lightAngle * 0.6f ) );
			const float3 rimLight  = hiprt::normalize( rotateAroundY( float3{ 0.8f, 0.4f, -0.25f }, lightAngle ) );

			hiprtRay shadowRay;
			shadowRay.origin	= surfacePt + 2.0e-3f * Ng;
			shadowRay.direction = toLight;
			shadowRay.maxT		= 1.0e6f;

			hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> shadowTraversal(
				scene, shadowRay, stack, instanceStack );
			const hiprtHit shadowHit = shadowTraversal.getNextHit();
			const float	 visibility = shadowHit.hasHit() ? 0.0f : 1.0f;

			const float key = visibility * max( 0.0f, hiprt::dot( Ns, toLight ) );
			const float fill = 0.22f * max( 0.0f, hiprt::dot( Ns, fillLight ) );
			const float rim = 0.30f * powf( max( 0.0f, hiprt::dot( Ns, rimLight ) ), 6.0f );
			const float ambient = 0.10f;

			finalColor = diffuseColor * ( ambient + 1.25f * key + fill ) + hiprt::make_float3( rim );
			finalColor = gammaCorrect( finalColor );
		}
	}

	image[index * 4 + 0] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.x ), 0u, 255u );
	image[index * 4 + 1] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.y ), 0u, 255u );
	image[index * 4 + 2] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.z ), 0u, 255u );
	image[index * 4 + 3] = 255;
}
