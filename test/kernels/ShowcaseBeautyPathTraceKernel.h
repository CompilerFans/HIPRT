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

struct BeautyMaterial
{
	float3 baseColor;
	float3 specularColor;
	float3 absorptionColor;
	float  metallic;
	float  roughness;
	float  transmission;
	float  ior;
	float  emissionScale;
};

HIPRT_DEVICE HIPRT_INLINE float beautyLuma( const float3& c ) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

HIPRT_DEVICE HIPRT_INLINE float3 beautyMul( const float3& a, const float3& b )
{
	return { a.x * b.x, a.y * b.y, a.z * b.z };
}

HIPRT_DEVICE HIPRT_INLINE float3 beautyClamp( const float3& c )
{
	return { hiprt::clamp( c.x, 0.0f, 8.0f ), hiprt::clamp( c.y, 0.0f, 8.0f ), hiprt::clamp( c.z, 0.0f, 8.0f ) };
}

HIPRT_DEVICE HIPRT_INLINE float3 beautyRotateY( const float3& v, float angle )
{
	const float s = sinf( angle );
	const float c = cosf( angle );
	return { c * v.x + s * v.z, v.y, -s * v.x + c * v.z };
}

HIPRT_DEVICE HIPRT_INLINE float3 beautyReflect( const float3& v, const float3& n ) { return v - 2.0f * hiprt::dot( v, n ) * n; }

HIPRT_DEVICE HIPRT_INLINE bool beautyRefract( const float3& v, const float3& n, float eta, float3& outDir )
{
	const float cosI = -hiprt::dot( n, v );
	const float sin2T = eta * eta * max( 0.0f, 1.0f - cosI * cosI );
	if ( sin2T > 1.0f ) return false;
	const float cosT = sqrtf( max( 0.0f, 1.0f - sin2T ) );
	outDir = hiprt::normalize( eta * v + ( eta * cosI - cosT ) * n );
	return true;
}

HIPRT_DEVICE HIPRT_INLINE float3 beautyAces( const float3& x )
{
	const float3 a = x * ( 2.51f * x + 0.03f );
	const float3 b = x * ( 2.43f * x + 0.59f ) + 0.14f;
	return { hiprt::clamp( a.x / b.x, 0.0f, 1.0f ), hiprt::clamp( a.y / b.y, 0.0f, 1.0f ), hiprt::clamp( a.z / b.z, 0.0f, 1.0f ) };
}

HIPRT_DEVICE HIPRT_INLINE float3 beautySky( const float3& dir, float angle )
{
	const float3 rotated = beautyRotateY( dir, angle * 0.35f );
	const float  t	   = hiprt::clamp( 0.5f * ( rotated.y + 1.0f ), 0.0f, 1.0f );
	const float3 low	   = float3{ 0.018f, 0.022f, 0.030f };
	const float3 high	   = float3{ 0.085f, 0.115f, 0.165f };
	float3	   color   = ( 1.0f - t ) * low + t * high;

	const float horizon = powf( 1.0f - fabs( rotated.y ), 6.0f );
	color += float3{ 0.08f, 0.05f, 0.03f } * horizon;

	const float3 glowDir = hiprt::normalize( beautyRotateY( float3{ -0.35f, 0.45f, -0.82f }, angle ) );
	const float  sunTerm = powf( max( 0.0f, hiprt::dot( rotated, glowDir ) ), 96.0f );
	color += float3{ 2.5f, 1.8f, 1.2f } * sunTerm;
	return color;
}

HIPRT_DEVICE HIPRT_INLINE BeautyMaterial classifyBeautyMaterial( const float3& diffuseColor, bool isEmitter )
{
	BeautyMaterial m{};
	const float		luma = beautyLuma( diffuseColor );

	if ( isEmitter )
	{
		m.baseColor		= diffuseColor;
		m.specularColor = diffuseColor;
		m.absorptionColor = hiprt::make_float3( 1.0f );
		m.metallic		= 0.0f;
		m.roughness		= 0.0f;
		m.transmission = 0.0f;
		m.ior		   = 1.0f;
		m.emissionScale = 6.0f;
		return m;
	}

	if ( diffuseColor.x < 0.28f && diffuseColor.y > 0.55f && diffuseColor.z > 0.70f )
	{
		m.baseColor		= diffuseColor;
		m.specularColor = float3{ 0.96f, 0.98f, 1.0f };
		m.absorptionColor = float3{ 0.82f, 0.96f, 1.0f };
		m.metallic		= 0.02f;
		m.roughness		= 0.02f;
		m.transmission = 0.96f;
		m.ior		   = 1.45f;
		m.emissionScale = 0.0f;
		return m;
	}

	if ( diffuseColor.x > 0.68f && diffuseColor.y > 0.70f && diffuseColor.z > 0.74f )
	{
		m.baseColor		= diffuseColor;
		m.specularColor = float3{ 0.97f, 0.97f, 0.98f };
		m.absorptionColor = hiprt::make_float3( 1.0f );
		m.metallic		= 0.96f;
		m.roughness		= 0.04f;
		m.transmission = 0.0f;
		m.ior		   = 1.0f;
		m.emissionScale = 0.0f;
		return m;
	}

	if ( luma > 0.55f )
	{
		m.baseColor		= diffuseColor * float3{ 1.02f, 1.0f, 0.95f };
		m.specularColor = float3{ 0.95f, 0.91f, 0.84f };
		m.absorptionColor = hiprt::make_float3( 1.0f );
		m.metallic		= 0.88f;
		m.roughness		= 0.16f;
		m.transmission = 0.0f;
		m.ior		   = 1.0f;
		m.emissionScale = 0.0f;
	}
	else if ( luma < 0.09f )
	{
		m.baseColor		= diffuseColor * float3{ 0.85f, 0.92f, 1.10f };
		m.specularColor = float3{ 0.14f, 0.16f, 0.18f };
		m.absorptionColor = hiprt::make_float3( 1.0f );
		m.metallic		= 0.05f;
		m.roughness		= 0.42f;
		m.transmission = 0.0f;
		m.ior		   = 1.0f;
		m.emissionScale = 0.0f;
	}
	else
	{
		m.baseColor		= diffuseColor * float3{ 0.85f, 0.9f, 1.0f };
		m.specularColor = float3{ 0.08f, 0.09f, 0.1f };
		m.absorptionColor = hiprt::make_float3( 1.0f );
		m.metallic		= 0.0f;
		m.roughness		= 0.68f;
		m.transmission = 0.0f;
		m.ior		   = 1.0f;
		m.emissionScale = 0.0f;
	}

	return m;
}

HIPRT_DEVICE HIPRT_INLINE float3 beautySampleAreaLight(
	const Light& light, const float3& x, float3& lightPoint, float3& lightNormal, float& pdf, const float2& xi )
{
	lightNormal	   = hiprt::cross( light.m_lv1 - light.m_lv0, light.m_lv2 - light.m_lv0 );
	const float area = sqrtf( hiprt::dot( lightNormal, lightNormal ) ) * 0.5f;
	lightNormal	   = hiprt::normalize( lightNormal );

	const float2 bary = make_float2( 1.0f - sqrtf( xi.x ), xi.y * sqrtf( xi.x ) );
	lightPoint		   = light.m_lv0 + bary.x * ( light.m_lv1 - light.m_lv0 ) + bary.y * ( light.m_lv2 - light.m_lv0 );

	const float3 r		  = lightPoint - x;
	const float	dist2	  = hiprt::dot( r, r );
	const float cosOnLight = fabs( hiprt::dot( lightNormal, -hiprt::normalize( r ) ) );
	if ( dist2 < 1.0e-6f || cosOnLight < 1.0e-6f || hiprt::dot( r, lightNormal ) > 0.0f )
	{
		pdf = 0.0f;
		return hiprt::make_float3( 0.0f );
	}

	pdf = dist2 * ( 1.0f / area ) / cosOnLight;
	return light.m_le;
}

extern "C" __global__ void __launch_bounds__( 64 ) ShowcaseBeautyPathTraceKernel(
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

	constexpr uint32_t Spp		 = 128;
	constexpr uint32_t MaxBounces = 5;

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
				radiance += beautyMul( throughput, beautySky( ray.direction, lightAngle ) );
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

			const uint32_t matOffset = matOffsetPerInstance[hit.instanceID] + hit.primID;
			const uint32_t matIndex  = matIndices[matOffset];
			const float3   diffuse   = matIndex == hiprtInvalidValue ? hiprt::make_float3( 0.8f ) : materials[matIndex].m_diffuse;
			const bool	   isLight   = matIndex != hiprtInvalidValue && materials[matIndex].light();
			const float3   emission  = matIndex == hiprtInvalidValue ? hiprt::make_float3( 0.0f ) : materials[matIndex].m_emission;
			const BeautyMaterial mat = classifyBeautyMaterial( diffuse, isLight );

			if ( isLight )
			{
				radiance += beautyMul( throughput, emission * mat.emissionScale );
				break;
			}

			const float3 surfacePt = ray.origin + hit.t * ray.direction;

			if ( numOfLights[0] > 0 )
			{
				const uint32_t lightIndex = min( static_cast<uint32_t>( randf( seed ) * numOfLights[0] ), numOfLights[0] - 1 );
				float3		   lightPoint;
				float3		   lightNormal;
				float		   pdf = 0.0f;
				const float3   le  = beautySampleAreaLight(
					 lights[lightIndex], surfacePt, lightPoint, lightNormal, pdf, make_float2( randf( seed ), randf( seed ) ) );

				if ( pdf > 0.0f )
				{
					hiprtRay shadowRay;
					shadowRay.origin	= surfacePt + 2.0e-3f * Ng;
					shadowRay.direction = hiprt::normalize( lightPoint - surfacePt );
					shadowRay.maxT		= 0.999f * sqrtf( hiprt::dot( lightPoint - surfacePt, lightPoint - surfacePt ) );

					hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> shadowTraversal(
						scene, shadowRay, stack, instanceStack );
					if ( !shadowTraversal.getNextHit().hasHit() )
					{
						const float nDotL = max( 0.0f, hiprt::dot( Ns, shadowRay.direction ) );
						const float3 diffuseTerm = ( 1.0f - mat.metallic ) * mat.baseColor * ( nDotL / hiprt::Pi );
						radiance += beautyMul( throughput, diffuseTerm * le / pdf ) * static_cast<float>( numOfLights[0] );
					}
				}
			}

			const float3 keyDir	  = hiprt::normalize( beautyRotateY( float3{ -0.48f, 0.74f, -0.46f }, lightAngle ) );
			const float  sunAmount = max( 0.0f, hiprt::dot( Ns, keyDir ) );
			if ( sunAmount > 0.0f )
			{
				hiprtRay sunShadowRay;
				sunShadowRay.origin	  = surfacePt + 2.0e-3f * Ng;
				sunShadowRay.direction = keyDir;
				sunShadowRay.maxT	  = 1.0e6f;

				hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> sunShadowTraversal(
					scene, sunShadowRay, stack, instanceStack );
				if ( !sunShadowTraversal.getNextHit().hasHit() )
				{
					const float3 sunColor = float3{ 1.8f, 1.35f, 0.9f };
					radiance += beautyMul( throughput, ( 1.0f - mat.metallic ) * mat.baseColor * ( 0.22f * sunAmount ) * sunColor );
				}
			}

			const float3 reflectDir = hiprt::normalize( beautyReflect( ray.direction, Ns ) );
			if ( mat.transmission > 0.0f )
			{
				const float3 outwardNormal = hiprt::normalize( hiprtVectorObjectToWorld( hit.normal, scene, hit.instanceID ) );
				const bool   entering	  = hiprt::dot( ray.direction, outwardNormal ) < 0.0f;
				const float3 refractNormal = entering ? outwardNormal : -outwardNormal;
				const float eta			  = entering ? ( 1.0f / mat.ior ) : mat.ior;
				const float cosTheta		  = hiprt::clamp( -hiprt::dot( ray.direction, refractNormal ), 0.0f, 1.0f );
				const float r0			  = ( 1.0f - mat.ior ) / ( 1.0f + mat.ior );
				const float fresnel		  = r0 * r0 + ( 1.0f - r0 * r0 ) * powf( 1.0f - cosTheta, 5.0f );

				float3 refractDir;
				const bool canRefract = beautyRefract( ray.direction, refractNormal, eta, refractDir );
				const bool chooseReflect = !canRefract || randf( seed ) < fresnel;
				if ( chooseReflect )
				{
					ray.origin = surfacePt + 2.0e-3f * Ng;
					ray.direction = reflectDir;
					throughput = beautyMul( throughput, mat.specularColor );
				}
				else
				{
					const float3 biasNormal = entering ? -refractNormal : refractNormal;
					ray.origin = surfacePt + 2.5e-3f * biasNormal;
					ray.direction = refractDir;
					throughput = beautyMul( throughput, mat.absorptionColor );
				}
				ray.maxT = 1.0e6f;
				continue;
			}

			const float  specMix	  = mat.metallic + 0.08f;
			const float3 diffuseDir = sampleHemisphereCosine( Ns, seed );
			const float3 glossyDir  = hiprt::normalize( reflectDir + mat.roughness * sampleHemisphereCosine( reflectDir, seed ) );
			const float3 nextDir	  = hiprt::normalize( ( 1.0f - specMix ) * diffuseDir + specMix * glossyDir );

			const float3 diffuseWeight  = ( 1.0f - mat.metallic ) * mat.baseColor;
			const float3 specularWeight = mat.specularColor;
			throughput = beautyMul( throughput, ( 1.0f - specMix ) * diffuseWeight + specMix * specularWeight );
			throughput = beautyClamp( throughput );

			if ( bounce >= 2 )
			{
				const float rr = hiprt::clamp( max( throughput.x, max( throughput.y, throughput.z ) ), 0.15f, 0.96f );
				if ( randf( seed ) > rr ) break;
				throughput = throughput * ( 1.0f / rr );
			}

			ray.origin	 = surfacePt + 2.0e-3f * Ng;
			ray.direction = nextDir;
			ray.maxT		 = 1.0e6f;
		}

		accumColor += radiance;
	}

	float3 finalColor = accumColor * ( 1.15f / static_cast<float>( Spp ) );
	finalColor		 = beautyAces( finalColor );
	finalColor		 = gammaCorrect( finalColor );

	const float2 uv = {
		( static_cast<float>( x ) + 0.5f ) / static_cast<float>( resolution.x ),
		( static_cast<float>( y ) + 0.5f ) / static_cast<float>( resolution.y ) };
	const float2 d   = uv - float2{ 0.5f, 0.5f };
	const float vignette = 1.0f - 0.75f * ( d.x * d.x + d.y * d.y );
	finalColor		   = beautyClamp( finalColor * vignette );

	image[index * 4 + 0] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.x ), 0u, 255u );
	image[index * 4 + 1] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.y ), 0u, 255u );
	image[index * 4 + 2] = hiprt::clamp( static_cast<uint32_t>( 255.0f * finalColor.z ), 0u, 255u );
	image[index * 4 + 3] = 255;
}
