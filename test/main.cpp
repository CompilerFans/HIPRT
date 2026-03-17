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

#include "hiprtTest.h"
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Header.h>
#include <test/CornellBox.h>
#include <contrib/argparse/argparse.h>
#include <numeric>
#include <memory>
#include <cassert>
#include <map>
#include <chrono>

struct SceneTransformDebugData
{
	float3	 m_frameScale;
	uint32_t m_frameIndex;
	uint32_t m_frameCount;
	uint32_t m_isStatic;
	uint32_t m_isIdentity;
};

///

///

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, AoRayEmbreeHairball )
{
	constexpr bool	Timings	 = true;
	constexpr float AoRadius = 0.5f;

	Camera camera = createCamera<TestCasesType::TestHairball>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/hairball/hairball.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitCustomBvhImport,
		Timings );
	render(
		"AoRayEmbreeHairball.png",
		getRootDir() / "test/kernels/AoRayKernel.h",
		"AoRayKernel",
		"AoRayHairball.png",
		Timings,
		AoRadius );
	deleteScene( m_scene );
}
#endif

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, AoRayHairball )
{
	constexpr bool	Timings	 = true;
	constexpr float AoRadius = 0.5f;

	Camera camera = createCamera<TestCasesType::TestHairball>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/hairball/hairball.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferBalancedBuild,
		Timings );
	render(
		"AoRayHairball.png",
		getRootDir() / "test/kernels/AoRayKernel.h",
		"AoRayKernel",
		"AoRayHairball.png",
		Timings,
		AoRadius );
	deleteScene( m_scene );
}
#endif

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, AoRayEmbreeBistro )
{
	constexpr bool	Timings	 = true;
	constexpr float AoRadius = 50.0f;

	Camera camera = createCamera<TestCasesType::TestBistro>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/bistro_full/Exterior/exterior.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitCustomBvhImport,
		Timings );
	render(
		"AoRayEmbreeBistro.png",
		getRootDir() / "test/kernels/AoRayKernel.h",
		"AoRayKernel",
		"AoRayBistro.png",
		Timings,
		AoRadius );
	deleteScene( m_scene );
}
#endif

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, AoRayTransformedBistro )
{
	constexpr bool	Timings	 = true;
	constexpr float AoRadius = 50.0f;

	Camera camera = createCamera<TestCasesType::TestBistro>();

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, 0.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	transform.rotation	  = { 0.0f, 1.0f, 0.0f, 0.5f };

	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/bistro_full/Exterior/exterior.obj",
		false,
		transform,
		hiprtBuildFlagBitPreferHighQualityBuild,
		Timings );
	render(
		"AoRayTransformedBistro.png",
		getRootDir() / "test/kernels/AoRayKernel.h",
		"AoRayKernel",
		"AoRayTransformedBistro.png",
		Timings,
		AoRadius );
	deleteScene( m_scene );
}
#endif

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, AoRayBistro )
{
	constexpr bool	Timings	 = true;
	constexpr float AoRadius = 50.0f;

	Camera camera = createCamera<TestCasesType::TestBistro>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/bistro_full/Exterior/exterior.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferHighQualityBuild,
		Timings );
	render(
		"AoRayBistro.png", getRootDir() / "test/kernels/AoRayKernel.h", "AoRayKernel", "AoRayBistro.png", Timings, AoRadius );
	deleteScene( m_scene );
}
#endif

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, PrimaryRayBistro )
{
	constexpr uint32_t Option	  = VisualizeColor;
	constexpr bool	   Timings	  = true;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestBistro>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/bistro_full/Exterior/exterior.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferFastBuild,
		Timings );
	render(
		"PrimaryRayBistro.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "PrimaryRayBistro.png", Timings );
	deleteScene( m_scene );
}
#endif

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, BvhFastBistro )
{
	constexpr uint32_t Option	  = VisualizeColor;
	constexpr bool	   Timings	  = true;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestBistro>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/bistro_full/Exterior/exterior.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferFastBuild,
		Timings );
	render(
		"BvhFastBistro.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "PrimaryRayBistro.png", Timings );
	deleteScene( m_scene );
}
#endif

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, BvhHighQBistro )
{
	constexpr uint32_t Option	  = VisualizeColor;
	constexpr bool	   Timings	  = true;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestBistro>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/bistro_full/Exterior/exterior.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferHighQualityBuild,
		Timings );
	render(
		"BvhHighQBistro.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "PrimaryRayBistro.png", Timings );
	deleteScene( m_scene );
}
#endif

#ifndef HIPRT_PUBLIC_REPO
TEST_F( PerformanceTestCases, BvhBalancedBistro )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );
	bool			   getTimings = true;

	Camera camera = createCamera<TestCasesType::TestBistro>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/lfs/bistro_full/Exterior/exterior.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferBalancedBuild,
		getTimings );
	render(
		"BvhBalancedBistro.png",
		getRootDir() / "test/kernels/PrimaryRayKernel.h",
		kernelName,
		"PrimaryRayBistro.png",
		getTimings );
	deleteScene( m_scene );
}
#endif

TEST_F( ObjTestCases, TranslateCornellBox )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, -5.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	transform.rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform );
	render(
		"TranslateCornellBox.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "TranslateCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, ScaleCornellBox )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, 0.0f };
	transform.scale		  = { 2.0f, 2.0f, 2.0f };
	transform.rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform );
	render( "ScaleCornellBox.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "ScaleCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, RotateCornellBox )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, -3.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	transform.rotation	  = { 0.0f, 1.0f, 0.0f, 0.5f };

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform );
	render( "RotateCornellBox.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "RotateCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, RotateCornellBoxSmallAngleNoRef )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, -3.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	transform.rotation	  = { 0.0f, 1.0f, 0.0f, 0.1f };

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform );
	render( std::nullopt, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, std::nullopt );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingShowcaseNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 8.8f, 0.15f, 17.8f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.46f };
	camera.m_fov		 = 48.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );
	render(
		outputDir / "metax_ray_tracing_lit.png",
		getRootDir() / "test/kernels/ShowcaseKeyLightKernel.h",
		"ShowcaseKeyLightKernel",
		std::nullopt,
		false,
		0.15f );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingTurntableNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase_turntable";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 0.0f, 0.15f, 18.5f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.0f };
	camera.m_fov		 = 48.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );

	constexpr uint32_t frameCount = 12;
	constexpr float	targetZ	   = 1.4f;
	constexpr float	radius	   = 17.5f;
	constexpr float	height	   = 0.15f;
	constexpr float	startAngle = -0.55f;
	constexpr float	endAngle   = 0.55f;

	for ( uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex )
	{
		const float t	 = frameCount > 1 ? static_cast<float>( frameIndex ) / static_cast<float>( frameCount - 1 ) : 0.0f;
		const float angle = startAngle + ( endAngle - startAngle ) * t;
		char		frameName[32];
		snprintf( frameName, sizeof( frameName ), "frame_%04u.png", frameIndex );

		m_camera.m_translation = { radius * sinf( angle ), height, targetZ + radius * cosf( angle ) };
		m_camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, angle };
		render(
			outputDir / frameName,
			getRootDir() / "test/kernels/ShowcaseKeyLightKernel.h",
			"ShowcaseKeyLightKernel",
			std::nullopt,
			false,
			-startAngle + angle );
	}

	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingPathTraceNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 8.8f, 0.15f, 17.8f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.46f };
	camera.m_fov		 = 48.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );
	render(
		outputDir / "metax_ray_tracing_pathtrace.png",
		getRootDir() / "test/kernels/ShowcasePathTraceKernel.h",
		"ShowcasePathTraceKernel",
		std::nullopt,
		false,
		0.15f );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingPathTraceTurntableNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase_turntable_pathtrace";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 0.0f, 0.15f, 18.5f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.0f };
	camera.m_fov		 = 48.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );

	constexpr uint32_t frameCount = 8;
	constexpr float	targetZ	   = 1.4f;
	constexpr float	radius	   = 17.5f;
	constexpr float	height	   = 0.15f;
	constexpr float	startAngle = -0.42f;
	constexpr float	endAngle   = 0.42f;

	for ( uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex )
	{
		const float t	 = frameCount > 1 ? static_cast<float>( frameIndex ) / static_cast<float>( frameCount - 1 ) : 0.0f;
		const float angle = startAngle + ( endAngle - startAngle ) * t;
		char		frameName[32];
		snprintf( frameName, sizeof( frameName ), "frame_%04u.png", frameIndex );

		m_camera.m_translation = { radius * sinf( angle ), height, targetZ + radius * cosf( angle ) };
		m_camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, angle };
		render(
			outputDir / frameName,
			getRootDir() / "test/kernels/ShowcasePathTraceKernel.h",
			"ShowcasePathTraceKernel",
			std::nullopt,
			false,
			angle );
	}

	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingBeautyNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 7.6f, 0.05f, 16.2f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.50f };
	camera.m_fov		 = 44.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );
	render(
		outputDir / "metax_ray_tracing_beauty.png",
		getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
		"ShowcaseBeautyPathTraceKernel",
		std::nullopt,
		false,
		0.22f );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingBeautyTurntableNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase_turntable_beauty";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 0.0f, 0.05f, 16.8f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.0f };
	camera.m_fov		 = 44.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );

	constexpr uint32_t frameCount = 8;
	constexpr float	targetZ	   = 1.1f;
	constexpr float	radius	   = 15.8f;
	constexpr float	height	   = 0.05f;
	constexpr float	startAngle = -0.36f;
	constexpr float	endAngle   = 0.36f;

	for ( uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex )
	{
		const float t	 = frameCount > 1 ? static_cast<float>( frameIndex ) / static_cast<float>( frameCount - 1 ) : 0.0f;
		const float angle = startAngle + ( endAngle - startAngle ) * t;
		char		frameName[32];
		snprintf( frameName, sizeof( frameName ), "frame_%04u.png", frameIndex );

		m_camera.m_translation = { radius * sinf( angle ), height, targetZ + radius * cosf( angle ) };
		m_camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, angle };
		render(
			outputDir / frameName,
			getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
			"ShowcaseBeautyPathTraceKernel",
			std::nullopt,
			false,
			0.22f + angle );
	}

	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingPosterNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 6.1f, -0.35f, 13.6f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.57f };
	camera.m_fov		 = 36.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );
	render(
		outputDir / "metax_ray_tracing_poster.png",
		getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
		"ShowcaseBeautyPathTraceKernel",
		std::nullopt,
		false,
		0.28f );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingFinalPosterNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 5.4f, -0.42f, 12.2f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.61f };
	camera.m_fov		 = 32.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );
	render(
		outputDir / "metax_ray_tracing_final_poster.png",
		getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
		"ShowcaseBeautyPathTraceKernel",
		std::nullopt,
		false,
		0.32f );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingPosterTurntableNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase_turntable_poster";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 0.0f, -0.20f, 13.2f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.0f };
	camera.m_fov		 = 34.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_ray_tracing_stacked.obj" );

	constexpr uint32_t frameCount = 10;
	constexpr float	targetZ	   = 0.95f;
	constexpr float	radius	   = 12.5f;
	constexpr float	height	   = -0.20f;
	constexpr float	startAngle = -0.28f;
	constexpr float	endAngle   = 0.28f;

	for ( uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex )
	{
		const float t	 = frameCount > 1 ? static_cast<float>( frameIndex ) / static_cast<float>( frameCount - 1 ) : 0.0f;
		const float angle = startAngle + ( endAngle - startAngle ) * t;
		char		frameName[32];
		snprintf( frameName, sizeof( frameName ), "frame_%04u.png", frameIndex );

		m_camera.m_translation = { radius * sinf( angle ), height, targetZ + radius * cosf( angle ) };
		m_camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, angle };
		render(
			outputDir / frameName,
			getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
			"ShowcaseBeautyPathTraceKernel",
			std::nullopt,
			false,
			0.30f + angle );
	}

	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingGlassSceneNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 6.8f, -0.22f, 13.8f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.52f };
	camera.m_fov		 = 34.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_reflect_refract.obj" );
	render(
		outputDir / "metax_ray_tracing_glass.png",
		getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
		"ShowcaseBeautyPathTraceKernel",
		std::nullopt,
		false,
		0.34f );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingGlassTurntableNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase_turntable_glass";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 0.0f, -0.18f, 13.4f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.0f };
	camera.m_fov		 = 34.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_reflect_refract.obj" );

	constexpr uint32_t frameCount = 10;
	constexpr float	targetZ	   = 1.0f;
	constexpr float	radius	   = 12.6f;
	constexpr float	height	   = -0.18f;
	constexpr float	startAngle = -0.30f;
	constexpr float	endAngle   = 0.30f;

	for ( uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex )
	{
		const float t	 = frameCount > 1 ? static_cast<float>( frameIndex ) / static_cast<float>( frameCount - 1 ) : 0.0f;
		const float angle = startAngle + ( endAngle - startAngle ) * t;
		char		frameName[32];
		snprintf( frameName, sizeof( frameName ), "frame_%04u.png", frameIndex );

		m_camera.m_translation = { radius * sinf( angle ), height, targetZ + radius * cosf( angle ) };
		m_camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, angle };
		render(
			outputDir / frameName,
			getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
			"ShowcaseBeautyPathTraceKernel",
			std::nullopt,
			false,
			0.34f + angle );
	}

	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXRayTracingGlassLongAnimationNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase_turntable_glass_long";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 0.0f, -0.12f, 13.2f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.0f };
	camera.m_fov		 = 34.0f * hiprt::Pi / 180.0f;

	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/metax_reflect_refract.obj" );

	constexpr uint32_t frameCount = 120;
	constexpr float	targetZ	   = 1.0f;
	constexpr float	baseRadius  = 12.4f;
	constexpr float	baseHeight  = -0.12f;
	constexpr float	startAngle = -0.42f;
	constexpr float	endAngle   = 0.42f;

	for ( uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex )
	{
		const float t	 = frameCount > 1 ? static_cast<float>( frameIndex ) / static_cast<float>( frameCount - 1 ) : 0.0f;
		const float phase = t * hiprt::TwoPi;
		const float angle = startAngle + ( endAngle - startAngle ) * t;
		const float radius = baseRadius + 0.35f * sinf( phase * 0.5f );
		const float height = baseHeight + 0.10f * sinf( phase * 0.75f );
		const float lightAngle = 0.34f + 0.32f * sinf( phase + 0.35f ) + 0.10f * cosf( phase * 2.0f );
		char		frameName[32];
		snprintf( frameName, sizeof( frameName ), "frame_%04u.png", frameIndex );

		m_camera.m_translation = { radius * sinf( angle ), height, targetZ + radius * cosf( angle ) };
		m_camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, angle };
		render(
			outputDir / frameName,
			getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
			"ShowcaseBeautyPathTraceKernel",
			std::nullopt,
			false,
			lightAngle );
	}

	deleteScene( m_scene );
}

TEST_F( ObjTestCases, MetaXCornellBeautyNoRef )
{
	const std::filesystem::path outputDir = getRootDir() / "docs/images/showcase";
	std::filesystem::create_directories( outputDir );

	Camera camera;
	camera.m_translation = { 0.15f, 1.85f, 3.95f };
	camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, 0.02f };
	camera.m_fov		 = 34.0f * hiprt::Pi / 180.0f;
	setupScene( camera, getRootDir() / "test/common/meshes/metax_showcase/cornell_metax.obj" );
	render(
		outputDir / "metax_cornell_beauty.png",
		getRootDir() / "test/kernels/ShowcaseBeautyPathTraceKernel.h",
		"ShowcaseBeautyPathTraceKernel",
		std::nullopt,
		false,
		0.0f );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, RecreateCornellBoxTwiceNoRef )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );
	Camera			   camera	  = createCamera<TestCasesType::TestCornellBox>();

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, -3.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	transform.rotation	  = { 0.0f, 1.0f, 0.0f, 0.0f };

	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform );
	render( std::nullopt, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, std::nullopt );
	deleteScene( m_scene );

	transform.rotation = { 0.0f, 1.0f, 0.0f, 0.1f };
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform );
	render( std::nullopt, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, std::nullopt );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, RecreateCornellBoxTwiceSameTransformNoRef )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );
	Camera			   camera	  = createCamera<TestCasesType::TestCornellBox>();

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, -3.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	transform.rotation	  = { 0.0f, 1.0f, 0.0f, 0.1f };

	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform );
	render( std::nullopt, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, std::nullopt );
	deleteScene( m_scene );

	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform );
	render( std::nullopt, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, std::nullopt );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, RecreateCornellBoxTwiceBuildOnly )
{
	Camera camera = createCamera<TestCasesType::TestCornellBox>();

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, -3.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	transform.rotation	  = { 0.0f, 1.0f, 0.0f, 0.1f };

	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform, hiprtBuildFlagBitPreferFastBuild );
	deleteScene( m_scene );

	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform, hiprtBuildFlagBitPreferFastBuild );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, RenderCornellBoxTwiceSameSceneNoRef )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );
	Camera			   camera	  = createCamera<TestCasesType::TestCornellBox>();

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, -3.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	transform.rotation	  = { 0.0f, 1.0f, 0.0f, 0.1f };

	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform, hiprtBuildFlagBitPreferFastBuild );
	render( std::nullopt, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, std::nullopt );
	render( std::nullopt, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, std::nullopt );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, PrimaryRayKernelRecreateStableRegs )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );
	Camera			   camera	  = createCamera<TestCasesType::TestCornellBox>();

	auto buildRegs = [&]( float angle ) {
		hiprtFrameSRT transform;
		transform.translation = { 0.0f, 0.0f, -3.0f };
		transform.scale		  = { 1.0f, 1.0f, 1.0f };
		transform.rotation	  = { 0.0f, 1.0f, 0.0f, angle };

		setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj", false, transform, hiprtBuildFlagBitPreferFastBuild );
		cudaFunction_t func;
		std::string	   blockSizeDef		  = "-DBLOCK_SIZE=64";
		std::string	   sharedStackSizeDef = "-DSHARED_STACK_SIZE=16";
		std::vector<const char*> opts{ blockSizeDef.c_str(), sharedStackSizeDef.c_str() };
		buildTraceKernel( m_scene.m_ctx, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, func, opts );
		int numRegs{};
		checkOro( cuFuncGetAttribute( &numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, func ) );
		deleteScene( m_scene );
		return numRegs;
	};

	const int regs0 = buildRegs( 0.1f );
	const int regs1 = buildRegs( 0.1f );
	EXPECT_EQ( regs0, regs1 );
}

TEST_F( ObjTestCases, BvhFastCornellBox )
{
	constexpr uint32_t Option	  = VisualizeColor;
	constexpr bool	   Timings	  = true;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferFastBuild,
		Timings );
	render(
		"BvhFastCornellBox.png",
		getRootDir() / "test/kernels/PrimaryRayKernel.h",
		kernelName,
		"PrimaryRayCornellBox.png",
		Timings );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, BvhHighQCornellBox )
{
	constexpr uint32_t Option	  = VisualizeColor;
	constexpr bool	   Timings	  = true;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferHighQualityBuild,
		Timings );
	render(
		"BvhHighQCornellBox.png",
		getRootDir() / "test/kernels/PrimaryRayKernel.h",
		kernelName,
		"PrimaryRayCornellBox.png",
		Timings );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, ShadowRayCornellBox )
{
	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj" );
	render(
		"ShadowRayCornellBox.png",
		getRootDir() / "test/kernels/ShadowRayKernel.h",
		"ShadowRayKernel",
		"ShadowRayCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, AoRayCornellBox )
{
	constexpr bool	Timings	 = true;
	constexpr float AoRadius = 1.4f;

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj" );
	render(
		"AoRayCornellBox.png",
		getRootDir() / "test/kernels/AoRayKernel.h",
		"AoRayKernel",
		"AoRayCornellBox.png",
		Timings,
		AoRadius );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, AoRayEmbreeCornellBox )
{
	constexpr bool	Timings	 = true;
	constexpr float AoRadius = 1.4f;

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitCustomBvhImport );
	render(
		"AoRayEmbreeCornellBox.png",
		getRootDir() / "test/kernels/AoRayKernel.h",
		"AoRayKernel",
		"AoRayCornellBox.png",
		Timings,
		AoRadius );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, UvsCornellBox )
{
	constexpr uint32_t Option	  = VisualizeUv;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj" );
	render( "UvsCornellBox.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "UvsCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, PrimIdsCornellBox )
{
	constexpr uint32_t Option	  = VisualizeId;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj" );
	render( "PrimIdsCornellBox.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "PrimIdsCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, HitDistCornellBox )
{
	constexpr uint32_t Option	  = VisualizeHitDist;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj" );
	render( "HitDistCornellBox.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "HitDistCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, NormalsCornellBox )
{
	constexpr uint32_t Option	  = VisualizeNormal;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj" );
	render( "NormalsCornellBox.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "NormalsCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, BvhUpdateCornellBox )
{
	constexpr uint32_t Option	  = VisualizeColor;
	constexpr uint32_t FrameCount = 7;
	constexpr bool	   Timings	  = true;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	hiprtFrameSRT transform;
	transform.translation = { 0.0f, 0.0f, -3.0f };
	transform.scale		  = { 1.0f, 1.0f, 1.0f };
	Camera camera		  = createCamera<TestCasesType::TestCornellBox>();

	float angle = 0.0f;
	for ( uint32_t i = 0; i < FrameCount; i++, angle += 0.1 )
	{
		transform.rotation = { 0.0f, 1.0f, 0.0f, angle };
		setupScene(
			camera,
			getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj",
			false,
			transform,
			hiprtBuildFlagBitPreferFastBuild,
			Timings );
		render( std::nullopt, getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, std::nullopt, Timings );
		deleteScene( m_scene );
	}

	transform.rotation = { 0.0f, 1.0f, 0.0f, angle };
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj",
		false,
		transform,
		hiprtBuildFlagBitPreferFastBuild,
		Timings );
	render(
		"BvhUpdateCornellBox.png",
		getRootDir() / "test/kernels/PrimaryRayKernel.h",
		kernelName,
		"BvhUpdateCornellBox.png",
		Timings );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, BvhBalancedCornellBox )
{
	constexpr uint32_t Option	  = VisualizeColor;
	constexpr bool	   Timings	  = true;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene(
		camera,
		getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj",
		false,
		std::nullopt,
		hiprtBuildFlagBitPreferBalancedBuild,
		Timings );
	render(
		"BvhBalancedCornellBox.png",
		getRootDir() / "test/kernels/PrimaryRayKernel.h",
		kernelName,
		"PrimaryRayCornellBox.png",
		Timings );
	deleteScene( m_scene );
}

TEST_F( ObjTestCases, PrimaryRayCornellBox )
{
	constexpr uint32_t Option	  = VisualizeColor;
	const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( Option );

	Camera camera = createCamera<TestCasesType::TestCornellBox>();
	setupScene( camera, getRootDir() / "test/common/meshes/cornellbox/cornellBox.obj" );
	render(
		"PrimaryRayCornellBox.png", getRootDir() / "test/kernels/PrimaryRayKernel.h", kernelName, "PrimaryRayCornellBox.png" );
	deleteScene( m_scene );
}

TEST_F( hiprtTest, CudaEnabled )
{
// This build now targets CUDA directly, so the only hard requirement is that the
// CUDA toolkit include path is injected for runtime NVRTC compilation.
#ifndef HIPRT_CUDA_INCLUDE_DIR
	printf( "This build is missing the CUDA toolkit include path.\n" );
	ASSERT_TRUE( false );
#endif
}

TEST_F( hiprtTest, MinimumCornellBox )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= CornellBoxTriangleCount;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	std::array<uint32_t, 3 * CornellBoxTriangleCount> idx;
	std::iota( idx.begin(), idx.end(), 0 );
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh.triangleCount );

	mesh.vertexCount  = 3 * mesh.triangleCount;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), const_cast<float3*>( cornellBoxVertices.data() ), mesh.vertexCount );

	const uint32_t GeomType = 2;

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;
	geomInput.geomType				 = GeomType;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtFuncNameSet funcName_unused;
	hiprtFuncNameSet funcNameSet;
	funcNameSet.filterFuncName = "duplicityFilter";
	// use 'funcNameSet' at slot 'GeomType'
	// for the previous slots, use empty functions.
	std::vector<hiprtFuncNameSet> funcNameSets = { funcName_unused, funcName_unused, funcNameSet };

	cudaFunction_t func;
	buildTraceKernel(
			ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "CornellBoxKernel", func, std::nullopt, funcNameSets, 3, 1 );

	hiprtFuncDataSet funcDataSet;
	hiprtFuncTable	 funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 3, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, GeomType, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	uint32_t* matIndices;
	malloc( matIndices, mesh.triangleCount );
	copyHtoD( matIndices, cornellBoxMatIndices.data(), mesh.triangleCount );

	float3* diffusColors;
	malloc( diffusColors, CornellBoxMaterialCount );
	copyHtoD( diffusColors, const_cast<float3*>( cornellBoxDiffuseColors.data() ), CornellBoxMaterialCount );

	void* args[] = { &geom, &dst, &funcTable, &res, &matIndices, &diffusColors };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "MinimumCornellBox.png", dst, "MinimumCornellBox.png" );

	free( matIndices );
	free( diffusColors );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, Compaction )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= CornellBoxTriangleCount;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	std::array<uint32_t, 3 * CornellBoxTriangleCount> idx;
	std::iota( idx.begin(), idx.end(), 0 );
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh.triangleCount );

	mesh.vertexCount  = 3 * mesh.triangleCount;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), const_cast<float3*>( cornellBoxVertices.data() ), mesh.vertexCount );

	const uint32_t GeomType = 2;

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;
	geomInput.geomType				 = GeomType;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );
	checkHiprt( hiprtCompactGeometry( ctxt, 0, geom, geom ) );

	hiprtFuncNameSet funcName_unused;
	hiprtFuncNameSet funcNameSet;
	funcNameSet.filterFuncName = "duplicityFilter";
	// use 'funcNameSet' at slot 'GeomType'
	// for the previous slots, use empty functions.
	std::vector<hiprtFuncNameSet> funcNameSets = { funcName_unused, funcName_unused, funcNameSet };

	cudaFunction_t func;
	buildTraceKernel(
			ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "CornellBoxKernel", func, std::nullopt, funcNameSets, 3, 1 );

	hiprtFuncDataSet funcDataSet;
	hiprtFuncTable	 funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 3, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, GeomType, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	uint32_t* matIndices;
	malloc( matIndices, mesh.triangleCount );
	copyHtoD( matIndices, cornellBoxMatIndices.data(), mesh.triangleCount );

	float3* diffusColors;
	malloc( diffusColors, CornellBoxMaterialCount );
	copyHtoD( diffusColors, const_cast<float3*>( cornellBoxDiffuseColors.data() ), CornellBoxMaterialCount );

	void* args[] = { &geom, &dst, &funcTable, &res, &matIndices, &diffusColors };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "Compaction.png", dst, "MinimumCornellBox.png" );

	free( matIndices );
	free( diffusColors );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, BatchCornellBox )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= CornellBoxTriangleCount;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	std::array<uint32_t, 3 * CornellBoxTriangleCount> idx;
	std::iota( idx.begin(), idx.end(), 0 );
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh.triangleCount );

	mesh.vertexCount  = 3 * mesh.triangleCount;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), const_cast<float3*>( cornellBoxVertices.data() ), mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;
	geomInput.geomType				 = 0;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags			   = hiprtBuildFlagBitPreferFastBuild;
	options.batchBuildMaxPrimCount = 64u;
	checkHiprt( hiprtGetGeometriesBuildTemporaryBufferSize( ctxt, 1, &geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry  geom;
	hiprtGeometry* geomAddrs = &geom;
	checkHiprt( hiprtCreateGeometries( ctxt, 1, &geomInput, options, &geomAddrs ) );
	checkHiprt( hiprtBuildGeometries( ctxt, hiprtBuildOperationBuild, 1, &geomInput, options, geomTemp, 0, &geom ) );

	hiprtFuncNameSet funcNameSet;
	funcNameSet.filterFuncName				   = "duplicityFilter";
	std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

	cudaFunction_t func;
	buildTraceKernel(
			ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "CornellBoxKernel", func, std::nullopt, funcNameSets, 1, 1 );

	hiprtFuncDataSet funcDataSet;
	hiprtFuncTable	 funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	uint32_t* matIndices;
	malloc( matIndices, mesh.triangleCount );
	copyHtoD( matIndices, cornellBoxMatIndices.data(), mesh.triangleCount );

	float3* diffusColors;
	malloc( diffusColors, CornellBoxMaterialCount );
	copyHtoD( diffusColors, const_cast<float3*>( cornellBoxDiffuseColors.data() ), CornellBoxMaterialCount );

	void* args[] = { &geom, &dst, &funcTable, &res, &matIndices, &diffusColors };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "BatchCornellBox.png", dst, "MinimumCornellBox.png" );

	free( matIndices );
	free( diffusColors );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometries( ctxt, 1, &geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, BatchGeometryCornellSweepBuildOnly )
{
	constexpr size_t GeomCount = 2;
	const std::array<uint32_t, 5> triangleCounts = { 2u, 4u, 8u, 16u, 32u };

	hiprtBuildOptions options{};
	options.buildFlags			   = hiprtBuildFlagBitPreferFastBuild;
	options.batchBuildMaxPrimCount = 64u;

	for ( uint32_t triangleCount : triangleCounts )
	{
		std::cout << "BatchGeometryCornellSweepBuildOnly triangleCount=" << triangleCount << std::endl;

		std::vector<std::vector<void*>> garbageCollectors( GeomCount );
		std::vector<hiprtGeometryBuildInput> geomInputs( GeomCount );
		for ( size_t i = 0; i < GeomCount; ++i )
		{
			hiprtTriangleMeshPrimitive mesh;
			createCornellTriangleMeshPrimitive( triangleCount, mesh, garbageCollectors[i] );
			geomInputs[i] = {};
			geomInputs[i].type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInputs[i].primitive.triangleMesh = mesh;
			geomInputs[i].geomType				 = 0;
		}

		buildBatchGeometriesBuildOnly( geomInputs, options );

		for ( std::vector<void*>& garbageCollector : garbageCollectors )
			destroyGarbage( garbageCollector );
	}
}

TEST_F( hiprtTest, BatchGeometryIndexedQuadStripSweepBuildOnly )
{
	constexpr size_t GeomCount = 2;
	const std::array<uint32_t, 5> quadCounts = { 1u, 2u, 4u, 8u, 16u };

	hiprtBuildOptions options{};
	options.buildFlags			   = hiprtBuildFlagBitPreferFastBuild;
	options.batchBuildMaxPrimCount = 64u;

	for ( uint32_t quadCount : quadCounts )
	{
		std::cout << "BatchGeometryIndexedQuadStripSweepBuildOnly triangleCount=" << ( 2 * quadCount ) << std::endl;

		std::vector<std::vector<void*>> garbageCollectors( GeomCount );
		std::vector<hiprtGeometryBuildInput> geomInputs( GeomCount );
		for ( size_t i = 0; i < GeomCount; ++i )
		{
			hiprtTriangleMeshPrimitive mesh;
			createIndexedQuadStripTriangleMeshPrimitive( quadCount, mesh, garbageCollectors[i] );
			geomInputs[i] = {};
			geomInputs[i].type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInputs[i].primitive.triangleMesh = mesh;
			geomInputs[i].geomType				 = 0;
		}

		buildBatchGeometriesBuildOnly( geomInputs, options );

		for ( std::vector<void*>& garbageCollector : garbageCollectors )
			destroyGarbage( garbageCollector );
	}
}

TEST_F( hiprtTest, BatchGeometryIndexedQuadStripPrePairedSweepBuildOnly )
{
	constexpr size_t GeomCount = 2;
	const std::array<uint32_t, 5> quadCounts = { 1u, 2u, 4u, 8u, 16u };

	hiprtBuildOptions options{};
	options.buildFlags			   = hiprtBuildFlagBitPreferFastBuild;
	options.batchBuildMaxPrimCount = 64u;

	for ( uint32_t quadCount : quadCounts )
	{
		std::cout << "BatchGeometryIndexedQuadStripPrePairedSweepBuildOnly triangleCount=" << ( 2 * quadCount ) << std::endl;

		std::vector<std::vector<void*>> garbageCollectors( GeomCount );
		std::vector<hiprtGeometryBuildInput> geomInputs( GeomCount );
		for ( size_t i = 0; i < GeomCount; ++i )
		{
			hiprtTriangleMeshPrimitive mesh;
			createIndexedQuadStripTriangleMeshPrimitive( quadCount, mesh, garbageCollectors[i] );
			attachSequentialTrianglePairs( quadCount, mesh, garbageCollectors[i] );
			geomInputs[i] = {};
			geomInputs[i].type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInputs[i].primitive.triangleMesh = mesh;
			geomInputs[i].geomType				 = 0;
		}

		buildBatchGeometriesBuildOnly( geomInputs, options );

		for ( std::vector<void*>& garbageCollector : garbageCollectors )
			destroyGarbage( garbageCollector );
	}
}

TEST_F( hiprtTest, SceneAabbSingletonSrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	float3 aabbMin{}, aabbMax{};
	checkHiprt( hiprtExportSceneAabb( ctxt, scene, aabbMin, aabbMax ) );

	EXPECT_NEAR( aabbMin.x, -0.125f, 1.0e-5f );
	EXPECT_NEAR( aabbMin.y, -0.125f, 1.0e-5f );
	EXPECT_NEAR( aabbMin.z, 0.0f, 1.0e-5f );
	EXPECT_NEAR( aabbMax.x, 0.125f, 1.0e-5f );
	EXPECT_NEAR( aabbMax.y, 0.125f, 1.0e-5f );
	EXPECT_NEAR( aabbMax.z, 0.0f, 1.0e-5f );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneAabbSingletonMatrixShear )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameMatrix matrix{};
	matrix.matrix[0][0]	  = 1.0f;
	matrix.matrix[1][1]	  = 1.0f;
	matrix.matrix[2][2]	  = 1.0f;
	matrix.matrix[0][1]	  = 1.0f;
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeMatrix;
	malloc( reinterpret_cast<hiprtFrameMatrix*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameMatrix*>( sceneInput.instanceFrames ), &matrix, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	float3 aabbMin{}, aabbMax{};
	checkHiprt( hiprtExportSceneAabb( ctxt, scene, aabbMin, aabbMax ) );

	EXPECT_NEAR( aabbMin.x, -0.5f, 1.0e-5f );
	EXPECT_NEAR( aabbMin.y, -0.25f, 1.0e-5f );
	EXPECT_NEAR( aabbMin.z, 0.0f, 1.0e-5f );
	EXPECT_NEAR( aabbMax.x, 0.5f, 1.0e-5f );
	EXPECT_NEAR( aabbMax.y, 0.25f, 1.0e-5f );
	EXPECT_NEAR( aabbMax.z, 0.0f, 1.0e-5f );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneSingletonSrtNodeUsesTransformHeader )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	hiprt::SceneHeader header{};
	copyDtoH( &header, reinterpret_cast<hiprt::SceneHeader*>( scene ), 1 );

	hiprt::Frame frameOut{};
	copyDtoH( &frameOut, header.m_frames, 1 );

	hiprt::UserInstanceNode instanceNode{};
	copyDtoH( &instanceNode, reinterpret_cast<hiprt::UserInstanceNode*>( header.m_primNodes ), 1 );

	EXPECT_EQ( header.m_primCount, 1u );
	EXPECT_EQ( header.m_primNodeCount, 1u );
	EXPECT_EQ( header.m_frameCount, 1u );
	EXPECT_NEAR( frameOut.m_scale.x, 0.5f, 1.0e-5f );
	EXPECT_NEAR( frameOut.m_scale.y, 0.5f, 1.0e-5f );
	EXPECT_NEAR( frameOut.m_scale.z, 0.5f, 1.0e-5f );
	EXPECT_EQ( instanceNode.m_static, 0u );
	EXPECT_EQ( instanceNode.m_identity, 0u );
	EXPECT_EQ( instanceNode.m_transform.frameIndex, 0u );
	EXPECT_EQ( instanceNode.m_transform.frameCount, 1u );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneWorldToObjectRaySrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "TransformRayKernel", func );

	hiprtRay ray{};
	ray.origin	  = { 0.0f, 0.0f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 1000.0f;

	hiprtRay* dOut;
	malloc( dOut, 1 );
	void* args[] = { &scene, &ray, &dOut };
	launchKernel( func, 1, 1, 1, 1, args );
	waitForCompletion();

	hiprtRay out{};
	copyDtoH( &out, dOut, 1 );
	EXPECT_NEAR( out.origin.x, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.origin.y, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.origin.z, -2.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.x, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.y, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.z, 2.0f, 1.0e-5f );

	free( dOut );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneInternalTransformRaySrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "InternalTransformRayKernel", func );

	hiprtRay ray{};
	ray.origin	  = { 0.0f, 0.0f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 1000.0f;

	hiprtRay* dOut;
	malloc( dOut, 1 );
	void* args[] = { &scene, &ray, &dOut };
	launchKernel( func, 1, 1, 1, 1, args );
	waitForCompletion();

	hiprtRay out{};
	copyDtoH( &out, dOut, 1 );
	EXPECT_NEAR( out.origin.x, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.origin.y, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.origin.z, -2.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.x, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.y, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.z, 2.0f, 1.0e-5f );

	free( dOut );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneTransformDebugSrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "SceneTransformDebugKernel", func );

	SceneTransformDebugData* dOut;
	malloc( dOut, 1 );
	void* args[] = { &scene, &dOut };
	launchKernel( func, 1, 1, 1, 1, args );
	waitForCompletion();

	SceneTransformDebugData out{};
	copyDtoH( &out, dOut, 1 );
	EXPECT_NEAR( out.m_frameScale.x, 0.5f, 1.0e-5f );
	EXPECT_NEAR( out.m_frameScale.y, 0.5f, 1.0e-5f );
	EXPECT_NEAR( out.m_frameScale.z, 0.5f, 1.0e-5f );
	EXPECT_EQ( out.m_frameIndex, 0u );
	EXPECT_EQ( out.m_frameCount, 1u );
	EXPECT_EQ( out.m_isStatic, 0u );
	EXPECT_EQ( out.m_isIdentity, 0u );

	free( dOut );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneInterpolatedFrameDebugSrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "SceneInterpolatedFrameDebugKernel", func );

	SceneTransformDebugData* dOut;
	malloc( dOut, 1 );
	void* args[] = { &scene, &dOut };
	launchKernel( func, 1, 1, 1, 1, args );
	waitForCompletion();

	SceneTransformDebugData out{};
	copyDtoH( &out, dOut, 1 );
	EXPECT_NEAR( out.m_frameScale.x, 0.5f, 1.0e-5f );
	EXPECT_NEAR( out.m_frameScale.y, 0.5f, 1.0e-5f );
	EXPECT_NEAR( out.m_frameScale.z, 0.5f, 1.0e-5f );
	EXPECT_EQ( out.m_frameCount, 1u );
	EXPECT_EQ( out.m_isIdentity, 0u );

	free( dOut );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneInverseMatrixDebugSrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "SceneInverseMatrixDebugKernel", func );

	hiprtFrameMatrix* dOut;
	malloc( dOut, 1 );
	void* args[] = { &scene, &dOut };
	launchKernel( func, 1, 1, 1, 1, args );
	waitForCompletion();

	hiprtFrameMatrix out{};
	copyDtoH( &out, dOut, 1 );
	EXPECT_NEAR( out.matrix[0][0], 2.0f, 1.0e-5f );
	EXPECT_NEAR( out.matrix[1][1], 2.0f, 1.0e-5f );
	EXPECT_NEAR( out.matrix[2][2], 2.0f, 1.0e-5f );
	EXPECT_NEAR( out.matrix[0][3], 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.matrix[1][3], 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.matrix[2][3], 0.0f, 1.0e-5f );

	free( dOut );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneWorldToObjectRayMatrixShear )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameMatrix matrix{};
	matrix.matrix[0][0]	  = 1.0f;
	matrix.matrix[1][1]	  = 1.0f;
	matrix.matrix[2][2]	  = 1.0f;
	matrix.matrix[0][1]	  = 1.0f;
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeMatrix;
	malloc( reinterpret_cast<hiprtFrameMatrix*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameMatrix*>( sceneInput.instanceFrames ), &matrix, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "TransformRayKernel", func );

	hiprtRay ray{};
	ray.origin	  = { 0.0f, 0.0f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 1000.0f;

	hiprtRay* dOut;
	malloc( dOut, 1 );
	void* args[] = { &scene, &ray, &dOut };
	launchKernel( func, 1, 1, 1, 1, args );
	waitForCompletion();

	hiprtRay out{};
	copyDtoH( &out, dOut, 1 );
	EXPECT_NEAR( out.origin.x, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.origin.y, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.origin.z, -1.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.x, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.y, 0.0f, 1.0e-5f );
	EXPECT_NEAR( out.direction.z, 1.0f, 1.0e-5f );

	free( dOut );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneClosestHitSingletonSrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "SceneClosestHitKernel", func );

	std::array<hiprtRay, 2> rays{};
	rays[0].origin	  = { 0.0f, 0.0f, -1.0f };
	rays[0].direction = { 0.0f, 0.0f, 1.0f };
	rays[0].maxT	  = 1000.0f;
	rays[1].origin	  = { 0.2f, 0.0f, -1.0f };
	rays[1].direction = { 0.0f, 0.0f, 1.0f };
	rays[1].maxT	  = 1000.0f;

	hiprtRay* dRays;
	hiprtHit* dHits;
	malloc( dRays, rays.size() );
	malloc( dHits, rays.size() );
	copyHtoD( dRays, rays.data(), rays.size() );

	std::array<hiprtHit, 2> initHits{};
	copyHtoD( dHits, initHits.data(), initHits.size() );

	uint32_t rayCount = static_cast<uint32_t>( rays.size() );
	void*	 args[]	 = { &scene, &rayCount, &dRays, &dHits };
	launchKernel( func, rayCount, 1, 1, 1, args );
	waitForCompletion();

	std::array<hiprtHit, 2> hits{};
	copyDtoH( hits.data(), dHits, hits.size() );
	EXPECT_TRUE( hits[0].hasHit() );
	EXPECT_FALSE( hits[1].hasHit() );

	free( dRays );
	free( dHits );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneClosestHitSingletonSrtRecreate )
{
	auto runOnce = [&]( float angle ) {
		hiprtContext ctxt;
		checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
		checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= 2;
		mesh.triangleStride = sizeof( uint3 );
		malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
		uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
		copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

		mesh.vertexCount  = 4;
		mesh.vertexStride = sizeof( float3 );
		malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
		constexpr float Offset = 0.25f;
		float3 v[] = {
			{ -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
		copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

		hiprtGeometryBuildInput geomInput{};
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

		hiprtGeometry geom;
		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

		hiprtInstance instance{};
		instance.type	  = hiprtInstanceTypeGeometry;
		instance.geometry = geom;

		hiprtSceneBuildInput sceneInput{};
		sceneInput.instanceCount			   = 1;
		sceneInput.instanceTransformHeaders = nullptr;
		sceneInput.instanceMasks			   = nullptr;
		malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
		copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

		hiprtFrameSRT frame{};
		frame.translation	  = { 0.0f, 0.0f, 0.0f };
		frame.scale			  = { 0.5f, 0.5f, 0.5f };
		frame.rotation		  = { 0.0f, 1.0f, 0.0f, angle };
		sceneInput.frameCount = 1;
		sceneInput.frameType  = hiprtFrameTypeSRT;
		malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
		copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

		size_t		   sceneTempSize;
		hiprtDevicePtr sceneTemp;
		checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

		hiprtScene scene;
		checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
		checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

		cudaFunction_t func;
		buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "SceneClosestHitKernel", func );

		std::array<hiprtRay, 2> rays{};
		rays[0].origin	  = { 0.0f, 0.0f, -1.0f };
		rays[0].direction = { 0.0f, 0.0f, 1.0f };
		rays[0].maxT	  = 1000.0f;
		rays[1].origin	  = { 0.2f, 0.0f, -1.0f };
		rays[1].direction = { 0.0f, 0.0f, 1.0f };
		rays[1].maxT	  = 1000.0f;

		hiprtRay* dRays;
		hiprtHit* dHits;
		malloc( dRays, rays.size() );
		malloc( dHits, rays.size() );
		copyHtoD( dRays, rays.data(), rays.size() );
		std::array<hiprtHit, 2> initHits{};
		copyHtoD( dHits, initHits.data(), initHits.size() );

		uint32_t rayCount = static_cast<uint32_t>( rays.size() );
		void*	 args[]	 = { &scene, &rayCount, &dRays, &dHits };
		launchKernel( func, rayCount, 1, 1, 1, args );
		waitForCompletion();

		std::array<hiprtHit, 2> hits{};
		copyDtoH( hits.data(), dHits, hits.size() );

		free( dRays );
		free( dHits );
		free( sceneInput.instances );
		free( sceneInput.instanceFrames );
		free( mesh.triangleIndices );
		free( mesh.vertices );
		free( sceneTemp );
		free( geomTemp );
		checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
		checkHiprt( hiprtDestroyScene( ctxt, scene ) );
		checkHiprt( hiprtDestroyContext( ctxt ) );

		return hits;
	};

	auto hits0 = runOnce( 0.0f );
	EXPECT_TRUE( hits0[0].hasHit() );
	EXPECT_FALSE( hits0[1].hasHit() );

	auto hits1 = runOnce( 0.1f );
	EXPECT_TRUE( hits1[0].hasHit() );
	EXPECT_FALSE( hits1[1].hasHit() );
}

TEST_F( hiprtTest, SceneManualClosestHitSingletonSrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "SceneManualClosestHitKernel", func );

	std::array<hiprtRay, 2> rays{};
	rays[0].origin	  = { 0.0f, 0.0f, -1.0f };
	rays[0].direction = { 0.0f, 0.0f, 1.0f };
	rays[0].maxT	  = 1000.0f;
	rays[1].origin	  = { 0.2f, 0.0f, -1.0f };
	rays[1].direction = { 0.0f, 0.0f, 1.0f };
	rays[1].maxT	  = 1000.0f;

	hiprtRay* dRays;
	hiprtHit* dHits;
	malloc( dRays, rays.size() );
	malloc( dHits, rays.size() );
	copyHtoD( dRays, rays.data(), rays.size() );
	std::array<hiprtHit, 2> initHits{};
	copyHtoD( dHits, initHits.data(), initHits.size() );

	uint32_t rayCount = static_cast<uint32_t>( rays.size() );
	void*	 args[]	 = { &scene, &rayCount, &dRays, &dHits };
	launchKernel( func, rayCount, 1, 1, 1, args );
	waitForCompletion();

	std::array<hiprtHit, 2> hits{};
	copyDtoH( hits.data(), dHits, hits.size() );
	EXPECT_TRUE( hits[0].hasHit() );
	EXPECT_FALSE( hits[1].hasHit() );

	free( dRays );
	free( dHits );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, GeomClosestHitScaledRay )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "GeomClosestHitKernel", func );

	std::array<hiprtRay, 2> rays{};
	rays[0].origin	  = { 0.0f, 0.0f, -2.0f };
	rays[0].direction = { 0.0f, 0.0f, 2.0f };
	rays[0].maxT	  = 1000.0f;
	rays[1].origin	  = { 0.2f, 0.0f, -2.0f };
	rays[1].direction = { 0.0f, 0.0f, 2.0f };
	rays[1].maxT	  = 1000.0f;

	hiprtRay* dRays;
	hiprtHit* dHits;
	malloc( dRays, rays.size() );
	malloc( dHits, rays.size() );
	copyHtoD( dRays, rays.data(), rays.size() );
	std::array<hiprtHit, 2> initHits{};
	copyHtoD( dHits, initHits.data(), initHits.size() );

	uint32_t rayCount = static_cast<uint32_t>( rays.size() );
	void*	 args[]	 = { &geom, &rayCount, &dRays, &dHits };
	launchKernel( func, rayCount, 1, 1, 1, args );
	waitForCompletion();

	std::array<hiprtHit, 2> hits{};
	copyDtoH( hits.data(), dHits, hits.size() );
	EXPECT_TRUE( hits[0].hasHit() );
	EXPECT_TRUE( hits[1].hasHit() );

	free( dRays );
	free( dHits );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneTraceKernelSingletonSrt )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = {
		{ -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount			   = 1;
	sceneInput.instanceTransformHeaders = nullptr;
	sceneInput.instanceMasks			   = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	constexpr uint32_t sharedStackSize	  = 16u;
	constexpr uint32_t blockWidth		  = 8u;
	constexpr uint32_t blockHeight		  = 8u;
	constexpr uint32_t blockSize		  = blockWidth * blockHeight;
	std::string		   blockSizeDef		  = "-DBLOCK_SIZE=" + std::to_string( blockSize );
	std::string		   sharedStackSizeDef = "-DSHARED_STACK_SIZE=" + std::to_string( sharedStackSize );
	std::vector<const char*> opts{ blockSizeDef.c_str(), sharedStackSizeDef.c_str() };

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "TraceKernel", func, opts );

	std::array<hiprtRay, 64> rays{};
	rays[0].origin	  = { 0.0f, 0.0f, -1.0f };
	rays[0].direction = { 0.0f, 0.0f, 1.0f };
	rays[0].maxT	  = 1000.0f;
	rays[1].origin	  = { 0.2f, 0.0f, -1.0f };
	rays[1].direction = { 0.0f, 0.0f, 1.0f };
	rays[1].maxT	  = 1000.0f;
	for ( size_t i = 2; i < rays.size(); ++i )
	{
		rays[i].origin	  = rays[1].origin;
		rays[i].direction = rays[1].direction;
		rays[i].maxT	  = rays[1].maxT;
	}

	hiprtRay* dRays;
	hiprtHit* dHits;
	malloc( dRays, rays.size() );
	malloc( dHits, rays.size() );
	copyHtoD( dRays, rays.data(), rays.size() );
	std::array<hiprtHit, 2> initHits{};
	copyHtoD( dHits, initHits.data(), initHits.size() );

	hiprtGlobalStackBufferInput stackInput{
		hiprtStackTypeGlobal, hiprtStackEntryTypeInteger, 64u, static_cast<uint32_t>( rays.size() ) };
	hiprtGlobalStackBuffer stackBuffer;
	checkHiprt( hiprtCreateGlobalStackBuffer( ctxt, stackInput, stackBuffer ) );

	uint32_t rayCount = static_cast<uint32_t>( rays.size() );
	void*	 args[]	 = { &scene, &rayCount, &stackBuffer, &dRays, &dHits };
	launchKernel( func, rayCount, 1, blockWidth, blockHeight, args );
	waitForCompletion();

	std::array<hiprtHit, 64> hits{};
	copyDtoH( hits.data(), dHits, hits.size() );
	EXPECT_TRUE( hits[0].hasHit() );
	EXPECT_FALSE( hits[1].hasHit() );

	checkHiprt( hiprtDestroyGlobalStackBuffer( ctxt, stackBuffer ) );
	free( dRays );
	free( dHits );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, BuildTraceKernelFromBitcode )
{
#if defined( HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND ) && HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND == 1
	if ( getEnvVariable( "HIPRT_FORCE_RUNTIME_BITCODE_TEST" ) != "1" )
	{
		std::cout << "Skip BuildTraceKernelFromBitcode on cu-bridge: runtime bitcode link still rejects external "
					 "mxcc/cu-bridge user binaries as invalid kernel image; precompiled workflow remains the supported path."
				  << std::endl;
		return;
	}
#endif

	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	constexpr uint32_t sharedStackSize	  = 16u;
	constexpr uint32_t blockWidth		  = 8u;
	constexpr uint32_t blockHeight		  = 8u;
	constexpr uint32_t blockSize		  = blockWidth * blockHeight;
	std::string		   blockSizeDef		  = "-DBLOCK_SIZE=" + std::to_string( blockSize );
	std::string		   sharedStackSizeDef = "-DSHARED_STACK_SIZE=" + std::to_string( sharedStackSize );
	std::vector<const char*> opts{ blockSizeDef.c_str(), sharedStackSizeDef.c_str() };

	cudaFunction_t func = nullptr;
	checkHiprt( buildTraceKernelFromBitcode(
		ctxt, getRootDir() / "test/bitcodes/runtime_bitcode_test.cu", "TraceKernel", func, opts ) );
	EXPECT_NE( func, nullptr );

	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, BuildTraceKernelWithMxccBundlePath )
{
	setenv( "HIPRT_EXTERNAL_DEVICE_COMPILER", "mxcc", 1 );

	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	std::string source;
	ASSERT_TRUE( readSourceCode( getRootDir() / "test/bitcodes/runtime_bitcode_test.cu", source ) );

	const char* functionName = "TraceKernel";
	std::vector<hiprtApiFunction> functions( 1 );
	checkHiprt( hiprtBuildTraceKernels(
		ctxt,
		1,
		&functionName,
		source.c_str(),
		"runtime_bitcode_test.cu",
		0,
		nullptr,
		nullptr,
		0,
		nullptr,
		0,
		1,
		nullptr,
		functions.data(),
		nullptr,
		false ) );
	EXPECT_NE( functions[0], nullptr );

	checkHiprt( hiprtDestroyContext( ctxt ) );
	unsetenv( "HIPRT_EXTERNAL_DEVICE_COMPILER" );
}

TEST_F( hiprtTest, BuildTraceKernelFromLinkedBundle )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	cudaFunction_t func = nullptr;
	checkHiprt( buildTraceKernelFromLinkedBundle(
		ctxt, getRootDir() / "test/bitcodes/runtime_bitcode_test.cu", "TraceKernel", func, true ) );
	EXPECT_NE( func, nullptr );

	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, BuildTraceKernelFromBitcodeWithCustomFuncTable )
{
#if defined( HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND ) && HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND == 1
	if ( getEnvVariable( "HIPRT_FORCE_RUNTIME_BITCODE_TEST" ) != "1" )
	{
		std::cout << "Skip BuildTraceKernelFromBitcodeWithCustomFuncTable on cu-bridge: runtime bitcode link still rejects "
					 "external mxcc/cu-bridge user binaries as invalid kernel image; precompiled workflow remains the "
					 "supported path."
				  << std::endl;
		return;
	}
#endif

	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtFuncNameSet funcName_unused{};
	hiprtFuncNameSet funcNameSet{};
	funcNameSet.filterFuncName = "cutoutFilter";
	std::vector<hiprtFuncNameSet> funcNameSets = { funcName_unused, funcName_unused, funcName_unused, funcNameSet };

	cudaFunction_t func = nullptr;
	checkHiprt( buildTraceKernelFromBitcode(
		ctxt,
		getRootDir() / "test/bitcodes/runtime_bitcode_test.cu",
		"CutoutKernel",
		func,
		std::nullopt,
		funcNameSets,
		4,
		1 ) );
	EXPECT_NE( func, nullptr );

	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, LoadPrecompiledTraceKernel )
{
	cudaFunction_t func = nullptr;
	CUmodule	   module = nullptr;
	checkHiprt( loadPrecompiledTraceKernel( "TraceKernel", func, &module ) );
	EXPECT_NE( func, nullptr );
	EXPECT_NE( module, nullptr );
	checkOro( cuModuleUnload( module ) );
}

TEST_F( hiprtTest, LoadPrecompiledTraceKernelWithCustomFuncTable )
{
	cudaFunction_t func = nullptr;
	CUmodule	   module = nullptr;
	checkHiprt( loadPrecompiledTraceKernel( "CutoutKernel", func, &module ) );
	EXPECT_NE( func, nullptr );
	EXPECT_NE( module, nullptr );
	checkOro( cuModuleUnload( module ) );
}

TEST_F( hiprtTest, LaunchPrecompiledTraceKernel )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh{};
	mesh.triangleCount	= 1;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 3;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	float3 v[] = { { -0.1f, -0.1f, 0.0f }, { -0.1f, 0.1f, 0.0f }, { 0.1f, -0.1f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options{};
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtInstance instance{};
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	hiprtSceneBuildInput sceneInput{};
	sceneInput.instanceCount = 1;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame{};
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 1.0f, 1.0f, 1.0f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeSRT;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	hiprtScene scene;
	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func   = nullptr;
	CUmodule	   module = nullptr;
	checkHiprt( loadPrecompiledTraceKernel( "TraceKernel", func, &module ) );

	std::array<hiprtRay, 64> rays{};
	rays[0].origin	  = { 0.0f, 0.0f, -1.0f };
	rays[0].direction = { 0.0f, 0.0f, 1.0f };
	rays[0].maxT	  = 1000.0f;
	rays[1].origin	  = { 0.3f, 0.0f, -1.0f };
	rays[1].direction = { 0.0f, 0.0f, 1.0f };
	rays[1].maxT	  = 1000.0f;
	for ( size_t i = 2; i < rays.size(); ++i )
		rays[i] = rays[1];

	hiprtRay* dRays = nullptr;
	hiprtHit* dHits = nullptr;
	malloc( dRays, rays.size() );
	malloc( dHits, rays.size() );
	copyHtoD( dRays, rays.data(), rays.size() );
	std::array<hiprtHit, 64> initHits{};
	copyHtoD( dHits, initHits.data(), initHits.size() );

	hiprtGlobalStackBufferInput stackInput{
		hiprtStackTypeGlobal, hiprtStackEntryTypeInteger, 64u, static_cast<uint32_t>( rays.size() ) };
	hiprtGlobalStackBuffer stackBuffer;
	checkHiprt( hiprtCreateGlobalStackBuffer( ctxt, stackInput, stackBuffer ) );

	constexpr uint32_t blockWidth	  = 8u;
	constexpr uint32_t blockHeight	  = 8u;
	constexpr uint32_t blockSize	  = blockWidth * blockHeight;
	uint32_t		   rayCount		  = static_cast<uint32_t>( rays.size() );
	void*			   args[]		  = { &scene, &rayCount, &stackBuffer, &dRays, &dHits };
	launchKernel( func, rayCount, 1, blockWidth, blockSize / blockWidth, args );
	waitForCompletion();

	std::array<hiprtHit, 64> hits{};
	copyDtoH( hits.data(), dHits, hits.size() );
	EXPECT_TRUE( hits[0].hasHit() );
	EXPECT_FALSE( hits[1].hasHit() );

	checkHiprt( hiprtDestroyGlobalStackBuffer( ctxt, stackBuffer ) );
	free( dRays );
	free( dHits );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	checkOro( cuModuleUnload( module ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, LaunchPrecompiledTraceKernelWithCustomFuncTable )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh{};
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	float3 v[] = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	const uint32_t GeomType = 3;

	hiprtGeometryBuildInput geomInput{};
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;
	geomInput.geomType				 = GeomType;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options{};
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	cudaFunction_t func   = nullptr;
	CUmodule	   module = nullptr;
	checkHiprt( loadPrecompiledTraceKernel( "CutoutKernel", func, &module ) );

	hiprtFuncDataSet funcDataSet{};
	hiprtFuncTable	 funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 4, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, GeomType, 0, funcDataSet ) );

	uint8_t* dst = nullptr;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };
	checkOro( cuMemsetD8( reinterpret_cast<size_t>( dst ), 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 ) );

	void* args[] = { &geom, &dst, &funcTable, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	waitForCompletion();

	std::vector<uint8_t> image( g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	copyDtoH( image.data(), dst, image.size() );
	bool hasHitPixel = false;
	for ( size_t i = 0; i < image.size(); i += 4 )
	{
		if ( image[i] != 0 || image[i + 1] != 0 || image[i + 2] != 0 )
		{
			hasHitPixel = true;
			break;
		}
	}
	EXPECT_TRUE( hasHitPixel );

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkOro( cuModuleUnload( module ) );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, BoundingBox )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= CornellBoxTriangleCount;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	std::array<uint32_t, 3 * CornellBoxTriangleCount> idx;
	std::iota( idx.begin(), idx.end(), 0 );
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh.triangleCount );

	mesh.vertexCount  = 3 * mesh.triangleCount;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), const_cast<float3*>( cornellBoxVertices.data() ), mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	float3 aabbMin, aabbMax;
	checkHiprt( hiprtExportGeometryAabb( ctxt, geom, aabbMin, aabbMax ) );

	printf(
		"Geometry bounding box: [%f %f %f] [%f %f %f]\n", aabbMin.x, aabbMin.y, aabbMin.z, aabbMax.x, aabbMax.y, aabbMax.z );

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, CustomBvhImport )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= CornellBoxTriangleCount;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	std::array<uint32_t, 3 * CornellBoxTriangleCount> idx;
	std::iota( idx.begin(), idx.end(), 0 );
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh.triangleCount );

	mesh.vertexCount  = 3 * mesh.triangleCount;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), const_cast<float3*>( cornellBoxVertices.data() ), mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;
	geomInput.geomType				 = 0;

	buildBvh( geomInput );

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitCustomBvhImport;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtFuncNameSet funcNameSet;
	funcNameSet.filterFuncName				   = "duplicityFilter";
	std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

	cudaFunction_t func;

	buildTraceKernel(
			ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "CornellBoxKernel", func, std::nullopt, funcNameSets, 1, 1 );

	hiprtFuncDataSet funcDataSet;
	hiprtFuncTable	 funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	uint32_t* matIndices;
	malloc( matIndices, mesh.triangleCount );
	copyHtoD( matIndices, cornellBoxMatIndices.data(), mesh.triangleCount );

	float3* diffusColors;
	malloc( diffusColors, CornellBoxMaterialCount );
	copyHtoD( diffusColors, const_cast<float3*>( cornellBoxDiffuseColors.data() ), CornellBoxMaterialCount );

	void* args[] = { &geom, &dst, &funcTable, &res, &matIndices, &diffusColors };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "CustomBvhImport.png", dst, "CustomBvhImport.png" );

	free( matIndices );
	free( diffusColors );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomInput.nodeList.leafNodes );
	free( geomInput.nodeList.internalNodes );
	free( dst );
	free( geomTemp );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, BvhIoApi )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= CornellBoxTriangleCount;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	std::array<uint32_t, 3 * CornellBoxTriangleCount> idx;
	std::iota( idx.begin(), idx.end(), 0 );
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh.triangleCount );

	mesh.vertexCount  = 3 * mesh.triangleCount;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), const_cast<float3*>( cornellBoxVertices.data() ), mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;
	geomInput.geomType				 = 0;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry inGeom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, inGeom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, inGeom ) );

	const char* filename = "geom.bin";
	checkHiprt( hiprtSaveGeometry( ctxt, inGeom, filename ) );

	hiprtGeometry outGeom;
	checkHiprt( hiprtLoadGeometry( ctxt, outGeom, filename ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, inGeom ) );

	cudaFunction_t		 func;
	hiprtFuncDataSet funcDataSet;
	hiprtFuncTable	 funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

	hiprtFuncNameSet funcNameSet;
	funcNameSet.filterFuncName				   = "duplicityFilter";
	std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

	buildTraceKernel(
			ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "CornellBoxKernel", func, std::nullopt, funcNameSets, 1, 1 );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	uint32_t* matIndices;
	malloc( matIndices, mesh.triangleCount );
	copyHtoD( matIndices, cornellBoxMatIndices.data(), mesh.triangleCount );

	float3* diffusColors;
	malloc( diffusColors, CornellBoxMaterialCount );
	copyHtoD( diffusColors, const_cast<float3*>( cornellBoxDiffuseColors.data() ), CornellBoxMaterialCount );

	void* args[] = { &outGeom, &dst, &funcTable, &res, &matIndices, &diffusColors };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "BvhIoApi.png", dst, "BvhIoApi.png" );

	free( matIndices );
	free( diffusColors );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, outGeom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, MeshIntersection )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 1;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 3;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	float3 v[] = { { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.5f, 1.0f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "MeshIntersectionKernel", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &geom, &dst, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "MeshIntersection.png", dst, "MeshIntersection.png" );

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, MeshIntersectionNonIndexed )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh{};
	mesh.triangleCount = 1;
	mesh.vertexCount   = 3;
	mesh.vertexStride  = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	float3 v[] = { { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.5f, 1.0f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "MeshIntersectionKernel", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &geom, &dst, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "MeshIntersectionNonIndexed.png", dst, "MeshIntersection.png" );

	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, PairTriangles )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 3;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 9;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	float3 v[] = {
		{ 0.0f, 0.0f, 1.0f },
		{ 1.0f, 0.0f, 1.0f },
		{ 0.5f, 1.0f, 1.0f },
		{ 0.0f, 0.0f, 0.5f },
		{ 1.0f, 0.0f, 0.5f },
		{ 0.5f, 1.0f, 0.5f },
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f },
		{ 0.5f, 1.0f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtScene			 scene;
	hiprtDevicePtr		 sceneTemp;
	hiprtSceneBuildInput sceneInput;

	hiprtInstance instance;
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	sceneInput.instanceCount			= 1;
	sceneInput.instanceMasks			= nullptr;
	sceneInput.instanceTransformHeaders = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame;
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 1.0f, 1.0f, 1.0f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t sceneTempSize;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "PairTrianglesKernel", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &scene, &dst, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "PairTriangles.png", dst, "PairTriangles.png" );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, Cutout )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	float3 v[] = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	const uint32_t GeomType = 3;

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;
	geomInput.geomType				 = GeomType;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtFuncNameSet funcName_unused;
	hiprtFuncNameSet funcNameSet;
	funcNameSet.filterFuncName = "cutoutFilter";
	// use 'funcNameSet' at slot 'GeomType'
	// for the previous slots, use empty functions.
	std::vector<hiprtFuncNameSet> funcNameSets = { funcName_unused, funcName_unused, funcName_unused, funcNameSet };

	cudaFunction_t func;
	buildTraceKernel(
			ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "CutoutKernel", func, std::nullopt, funcNameSets, 4, 1 );

	hiprtFuncDataSet funcDataSet;
	hiprtFuncTable	 funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 4, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, GeomType, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &geom, &dst, &funcTable, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "Cutout.png", dst, "Cutout.png" );

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, CustomIntersection )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtAABBListPrimitive list;
	list.aabbCount	= 3;
	list.aabbStride = 6 * sizeof( float );
	malloc( reinterpret_cast<float3*&>( list.aabbs ), 6 );

	float3 b[] = {
		{ 0.15f, 0.40f, 0.0f },
		{ 0.35f, 0.60f, 0.0f },
		{ 0.40f, 0.40f, 0.0f },
		{ 0.60f, 0.60f, 0.0f },
		{ 0.65f, 0.40f, 0.0f },
		{ 0.85f, 0.60f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( list.aabbs ), b, 6 );

	hiprtGeometryBuildInput geomInput;
	geomInput.type				 = hiprtPrimitiveTypeAABBList;
	geomInput.primitive.aabbList = list;
	geomInput.geomType			 = 0;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtFuncNameSet funcNameSet;
	funcNameSet.intersectFuncName			   = "intersectCircle";
	std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

	cudaFunction_t func;
	buildTraceKernel(
			ctxt,
			getRootDir() / "test/kernels/HiprtTestKernel.h",
			"CustomIntersectionKernel",
			func,
			std::nullopt,
			funcNameSets,
			1,
			1 );

	float* centers;
	malloc( centers, 3 );
	float h[] = { 0.25f, 0.5f, 0.75f };
	copyHtoD( centers, h, 3 );

	hiprtFuncDataSet funcDataSet;
	funcDataSet.intersectFuncData = centers;

	hiprtFuncTable funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &geom, &dst, &funcTable, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "CustomIntersection.png", dst, "CustomIntersection.png" );

	free( list.aabbs );
	free( geomTemp );
	free( centers );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneIntersectionSingleton )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 3, 4, 5 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 6;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );

	constexpr float Scale = 0.5f;
	float3			v[]	  = {
		   { 0.5f, Scale, 0.0f },
		   { 0.5f + Scale * 0.86f, -Scale * 0.5f, 0.0f },
		   { 0.5f - Scale * 0.86f, -Scale * 0.5f, 0.0f },
		   { -0.5f, Scale, 0.0f },
		   { -0.5f + Scale * 0.86f, -Scale * 0.5f, 0.0f },
		   { -0.5f - Scale * 0.86f, -Scale * 0.5f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtScene			 scene;
	hiprtDevicePtr		 sceneTemp;
	hiprtSceneBuildInput sceneInput;

	hiprtInstance instance;
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	sceneInput.instanceCount			= 1;
	sceneInput.instanceMasks			= nullptr;
	sceneInput.instanceTransformHeaders = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameSRT frame;
	frame.translation	  = { 0.0f, 0.0f, 0.0f };
	frame.scale			  = { 0.5f, 0.5f, 0.5f };
	frame.rotation		  = { 0.0f, 0.0f, 1.0f, 0.0f };
	sceneInput.frameCount = 1;
	malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), &frame, 1 );

	size_t sceneTempSize;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "SceneIntersectionSingleton", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &scene, &dst, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "SceneIntersectionSingleton.png", dst, "SceneIntersectionSingleton.png" );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( sceneTemp );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneIntersection )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtGeometry		   geomCircles;
	hiprtDevicePtr		   geomTempCircles;
	hiprtAABBListPrimitive list;
	{
		list.aabbCount	= 3;
		list.aabbStride = 2 * sizeof( float4 );
		malloc( reinterpret_cast<float4*&>( list.aabbs ), 6 );

		float4 b[] = {
			{ 0.15f, 0.40f, 1.0f, 0.0f },
			{ 0.35f, 0.60f, 1.0f, 0.0f },
			{ 0.40f, 0.40f, 1.0f, 0.0f },
			{ 0.60f, 0.60f, 1.0f, 0.0f },
			{ 0.65f, 0.40f, 1.0f, 0.0f },
			{ 0.85f, 0.60f, 1.0f, 0.0f } };
		copyHtoD( reinterpret_cast<float4*>( list.aabbs ), b, 6 );

		hiprtGeometryBuildInput geomInput;
		geomInput.type				 = hiprtPrimitiveTypeAABBList;
		geomInput.primitive.aabbList = list;
		geomInput.geomType			 = 0;

		size_t geomTempSize;

		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTempCircles ), geomTempSize );

		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geomCircles ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTempCircles, 0, geomCircles ) );
	}

	hiprtGeometry			   geomTris;
	hiprtDevicePtr			   geomTempTris;
	hiprtTriangleMeshPrimitive mesh;
	{
		mesh.triangleCount	= 3;
		mesh.triangleStride = sizeof( uint3 );
		malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
		std::vector<uint32_t> idx( 3 * mesh.triangleCount );
		std::iota( idx.begin(), idx.end(), 0 );
		copyHtoD(
			reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh.triangleCount );

		mesh.vertexCount  = 9;
		mesh.vertexStride = sizeof( float3 );
		malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
		float3 v[] = {
			{ 0.15f, 0.40f, 0.0f },
			{ 0.35f, 0.40f, 0.0f },
			{ 0.25f, 0.60f, 0.0f },
			{ 0.40f, 0.40f, 0.0f },
			{ 0.60f, 0.40f, 0.0f },
			{ 0.50f, 0.60f, 0.0f },
			{ 0.65f, 0.40f, 0.0f },
			{ 0.85f, 0.40f, 0.0f },
			{ 0.75f, 0.60f, 0.0f } };
		copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh;

		size_t			  geomTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTempTris ), geomTempSize );

		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geomTris ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTempTris, 0, geomTris ) );
	}

	hiprtScene			 scene;
	hiprtDevicePtr		 sceneTemp;
	hiprtSceneBuildInput sceneInput;
	{
		hiprtInstance instCircles;
		instCircles.type	 = hiprtInstanceTypeGeometry;
		instCircles.geometry = geomCircles;

		hiprtInstance instTris;
		instTris.type	  = hiprtInstanceTypeGeometry;
		instTris.geometry = geomTris;

		hiprtInstance instances[] = { instCircles, instTris };

		sceneInput.instanceCount			= 2;
		sceneInput.instanceMasks			= nullptr;
		sceneInput.instanceTransformHeaders = nullptr;
		malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
		copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), instances, sceneInput.instanceCount );

		constexpr float Offset = 0.05f;
		hiprtFrameSRT	frames[2];
		frames[0].translation = { 0.0f, Offset, 0.0f };
		frames[0].scale		  = { 1.0f, 1.0f, 1.0f };
		frames[0].rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[1].translation = { 0.0f, -Offset, 0.0f };
		frames[1].scale		  = { 1.0f, 1.0f, 1.0f };
		frames[1].rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };
		sceneInput.frameCount = 2;
		malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), sceneInput.frameCount );
		copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), frames, sceneInput.frameCount );

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

		checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
		checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );
	}

	hiprtFuncNameSet funcNameSet;
	funcNameSet.intersectFuncName			   = "intersectCircle";
	std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

	cudaFunction_t func;

	buildTraceKernel(
			ctxt,
			getRootDir() / "test/kernels/HiprtTestKernel.h",
			"SceneIntersectionKernel",
			func,
			std::nullopt,
			funcNameSets,
			1,
			1 );

	float* centers;
	malloc( centers, 3 );
	float h[] = { 0.25f, 0.5f, 0.75f };
	copyHtoD( centers, h, 3 );

	hiprtFuncDataSet funcDataSet;
	funcDataSet.intersectFuncData = centers;

	hiprtFuncTable funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &scene, &dst, &funcTable, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "SceneIntersection.png", dst, "SceneIntersection.png" );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.vertices );
	free( mesh.triangleIndices );
	free( list.aabbs );
	free( geomTempCircles );
	free( geomTempTris );
	free( sceneTemp );
	free( centers );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geomTris ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geomCircles ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, SceneIntersectionMlas )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtGeometry		   geomCircles;
	hiprtDevicePtr		   geomTempCircles;
	hiprtAABBListPrimitive list;
	{
		list.aabbCount	= 3;
		list.aabbStride = 2 * sizeof( float4 );
		malloc( reinterpret_cast<float4*&>( list.aabbs ), 6 );

		float4 b[] = {
			{ 0.15f, 0.40f, 1.0f, 0.0f },
			{ 0.35f, 0.60f, 1.0f, 0.0f },
			{ 0.40f, 0.40f, 1.0f, 0.0f },
			{ 0.60f, 0.60f, 1.0f, 0.0f },
			{ 0.65f, 0.40f, 1.0f, 0.0f },
			{ 0.85f, 0.60f, 1.0f, 0.0f } };
		copyHtoD( reinterpret_cast<float4*>( list.aabbs ), b, 6 );

		hiprtGeometryBuildInput geomInput;
		geomInput.type				 = hiprtPrimitiveTypeAABBList;
		geomInput.primitive.aabbList = list;
		geomInput.geomType			 = 0;

		size_t geomTempSize;

		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTempCircles ), geomTempSize );

		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geomCircles ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTempCircles, 0, geomCircles ) );
	}

	hiprtGeometry			   geomTris;
	hiprtDevicePtr			   geomTempTris;
	hiprtTriangleMeshPrimitive mesh;
	{
		mesh.triangleCount	= 3;
		mesh.triangleStride = sizeof( uint3 );
		malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
		std::vector<uint32_t> idx( 3 * mesh.triangleCount );
		std::iota( idx.begin(), idx.end(), 0 );
		copyHtoD(
			reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh.triangleCount );

		mesh.vertexCount  = 9;
		mesh.vertexStride = sizeof( float3 );
		malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
		float3 v[] = {
			{ 0.15f, 0.40f, 0.0f },
			{ 0.35f, 0.40f, 0.0f },
			{ 0.25f, 0.60f, 0.0f },
			{ 0.40f, 0.40f, 0.0f },
			{ 0.60f, 0.40f, 0.0f },
			{ 0.50f, 0.60f, 0.0f },
			{ 0.65f, 0.40f, 0.0f },
			{ 0.85f, 0.40f, 0.0f },
			{ 0.75f, 0.60f, 0.0f } };
		copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh;

		size_t			  geomTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTempTris ), geomTempSize );

		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geomTris ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTempTris, 0, geomTris ) );
	}

	hiprtScene			 sceneMid;
	hiprtDevicePtr		 sceneTempMid;
	hiprtSceneBuildInput sceneInputMid;
	{
		hiprtInstance instCircles;
		instCircles.type	 = hiprtInstanceTypeGeometry;
		instCircles.geometry = geomCircles;

		hiprtInstance instTris;
		instTris.type	  = hiprtInstanceTypeGeometry;
		instTris.geometry = geomTris;

		hiprtInstance instances[] = { instCircles, instTris };

		sceneInputMid.instanceCount			   = 2;
		sceneInputMid.instanceMasks			   = nullptr;
		sceneInputMid.instanceTransformHeaders = nullptr;
		malloc( reinterpret_cast<hiprtInstance*&>( sceneInputMid.instances ), sceneInputMid.instanceCount );
		copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInputMid.instances ), instances, sceneInputMid.instanceCount );

		constexpr float Offset = 0.05f;
		hiprtFrameSRT	frames[2];
		frames[0].translation	 = { 0.0f, Offset, 0.0f };
		frames[0].scale			 = { 1.0f, 1.0f, 1.0f };
		frames[0].rotation		 = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[1].translation	 = { 0.0f, -Offset, 0.0f };
		frames[1].scale			 = { 1.0f, 1.0f, 1.0f };
		frames[1].rotation		 = { 0.0f, 0.0f, 1.0f, 0.0f };
		sceneInputMid.frameCount = 2;
		malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInputMid.instanceFrames ), sceneInputMid.frameCount );
		copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInputMid.instanceFrames ), frames, sceneInputMid.frameCount );

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInputMid, options, sceneTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( sceneTempMid ), sceneTempSize );

		checkHiprt( hiprtCreateScene( ctxt, sceneInputMid, options, sceneMid ) );
		checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInputMid, options, sceneTempMid, 0, sceneMid ) );
	}

	hiprtScene			 sceneTop;
	hiprtDevicePtr		 sceneTempTop;
	hiprtSceneBuildInput sceneInputTop;
	{
		hiprtInstance instance;
		instance.type  = hiprtInstanceTypeScene;
		instance.scene = sceneMid;

		hiprtInstance instances[] = { instance, instance };

		sceneInputTop.instanceCount			   = 2;
		sceneInputTop.instanceMasks			   = nullptr;
		sceneInputTop.instanceTransformHeaders = nullptr;
		malloc( reinterpret_cast<hiprtInstance*&>( sceneInputTop.instances ), sceneInputTop.instanceCount );
		copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInputTop.instances ), instances, sceneInputTop.instanceCount );

		constexpr float Offset = 0.12f;
		hiprtFrameSRT	frames[2];
		frames[0].translation	 = { 0.0f, Offset, 0.0f };
		frames[0].scale			 = { 1.0f, 1.0f, 1.0f };
		frames[0].rotation		 = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[1].translation	 = { 0.0f, -Offset, 0.0f };
		frames[1].scale			 = { 1.0f, 1.0f, 1.0f };
		frames[1].rotation		 = { 0.0f, 0.0f, 1.0f, 0.0f };
		sceneInputTop.frameCount = 2;
		malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInputTop.instanceFrames ), sceneInputTop.frameCount );
		copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInputTop.instanceFrames ), frames, sceneInputTop.frameCount );

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInputTop, options, sceneTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( sceneTempTop ), sceneTempSize );

		checkHiprt( hiprtCreateScene( ctxt, sceneInputTop, options, sceneTop ) );
		checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInputTop, options, sceneTempTop, 0, sceneTop ) );
	}

	hiprtFuncNameSet funcNameSet;
	funcNameSet.intersectFuncName			   = "intersectCircle";
	std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

	cudaFunction_t func;
	buildTraceKernel(
			ctxt,
			getRootDir() / "test/kernels/HiprtTestKernel.h",
			"SceneIntersectionKernel",
			func,
			std::nullopt,
			funcNameSets,
			1,
			1 );

	float* centers;
	malloc( centers, 3 );
	float h[] = { 0.25f, 0.5f, 0.75f };
	copyHtoD( centers, h, 3 );

	hiprtFuncDataSet funcDataSet;
	funcDataSet.intersectFuncData = centers;

	hiprtFuncTable funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &sceneTop, &dst, &funcTable, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "SceneIntersectionMlas.png", dst, "SceneIntersectionMlas.png" );

	free( sceneInputMid.instances );
	free( sceneInputMid.instanceFrames );
	free( sceneInputTop.instances );
	free( sceneInputTop.instanceFrames );
	free( mesh.vertices );
	free( mesh.triangleIndices );
	free( list.aabbs );
	free( geomTempCircles );
	free( geomTempTris );
	free( sceneTempMid );
	free( sceneTempTop );
	free( centers );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geomTris ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geomCircles ) );
	checkHiprt( hiprtDestroyScene( ctxt, sceneMid ) );
	checkHiprt( hiprtDestroyScene( ctxt, sceneTop ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, Shear )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 0, 2, 3 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 4;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	constexpr float Offset = 0.25f;
	float3 v[] = { { -Offset, -Offset, 0.0f }, { -Offset, Offset, 0.0f }, { Offset, Offset, 0.0f }, { Offset, -Offset, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	hiprtScene			 scene;
	hiprtDevicePtr		 sceneTemp;
	hiprtSceneBuildInput sceneInput;

	hiprtInstance instance;
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	sceneInput.instanceCount			= 1;
	sceneInput.instanceMasks			= nullptr;
	sceneInput.instanceTransformHeaders = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameMatrix matrix{};
	matrix.matrix[0][0]	  = 1.0f;
	matrix.matrix[1][1]	  = 1.0f;
	matrix.matrix[2][2]	  = 1.0f;
	matrix.matrix[0][1]	  = 1.0f;
	sceneInput.frameCount = 1;
	sceneInput.frameType  = hiprtFrameTypeMatrix;
	malloc( reinterpret_cast<hiprtFrameMatrix*&>( sceneInput.instanceFrames ), 1 );
	copyHtoD( reinterpret_cast<hiprtFrameMatrix*>( sceneInput.instanceFrames ), &matrix, 1 );

	size_t sceneTempSize;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "SceneIntersectionSingleton", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &scene, &dst, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "Shear.png", dst, "Shear.png" );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( sceneTemp );
	free( dst );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, MotionBlur )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtGeometry			   geomTris0;
	hiprtDevicePtr			   geomTempTris0;
	hiprtTriangleMeshPrimitive mesh0;
	{
		mesh0.triangleCount	 = 1;
		mesh0.triangleStride = sizeof( uint3 );
		malloc( reinterpret_cast<uint3*&>( mesh0.triangleIndices ), mesh0.triangleCount );
		std::vector<uint32_t> idx( 3 * mesh0.triangleCount );
		std::iota( idx.begin(), idx.end(), 0 );
		copyHtoD(
			reinterpret_cast<uint3*>( mesh0.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh0.triangleCount );

		mesh0.vertexCount  = 3;
		mesh0.vertexStride = sizeof( float3 );
		malloc( reinterpret_cast<float3*&>( mesh0.vertices ), mesh0.vertexCount );
		constexpr float Scale = 0.15f;
		float3			v[]	  = {
			   { Scale * sinf( 0.0f ), Scale * cosf( 0.0f ), 0.0f },
			   { Scale * sinf( hiprt::Pi * 2.0f / 3.0f ), Scale * cosf( hiprt::Pi * 2.0f / 3.0f ), 0.0f },
			   { Scale * sinf( hiprt::Pi * 4.0f / 3.0f ), Scale * cosf( hiprt::Pi * 4.0f / 3.0f ), 0.0f } };
		copyHtoD( reinterpret_cast<float3*>( mesh0.vertices ), v, mesh0.vertexCount );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh0;

		size_t			  geomTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTempTris0 ), geomTempSize );

		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geomTris0 ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTempTris0, 0, geomTris0 ) );
	}

	hiprtGeometry			   geomTris1;
	hiprtDevicePtr			   geomTempTris1;
	hiprtTriangleMeshPrimitive mesh1;
	{
		mesh1.triangleCount	 = 1;
		mesh1.triangleStride = sizeof( uint3 );
		malloc( reinterpret_cast<uint3*&>( mesh1.triangleIndices ), mesh1.triangleCount );
		std::vector<uint32_t> idx( 3 * mesh1.triangleCount );
		std::iota( idx.begin(), idx.end(), 0 );
		copyHtoD(
			reinterpret_cast<uint3*>( mesh1.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh1.triangleCount );

		mesh1.vertexCount  = 3;
		mesh1.vertexStride = sizeof( float3 );
		malloc( reinterpret_cast<float3*&>( mesh1.vertices ), mesh1.vertexCount );
		constexpr float Scale = 0.15f;
		float3			v[]	  = {
			   { Scale * sinf( 0.0f ), Scale * cosf( 0.0f ), 0.0f },
			   { Scale * sinf( hiprt::Pi * 2.0f / 3.0f ), Scale * cosf( hiprt::Pi * 2.0f / 3.0f ), 0.0f },
			   { Scale * sinf( hiprt::Pi * 4.0f / 3.0f ), Scale * cosf( hiprt::Pi * 4.0f / 3.0f ), 0.0f } };
		copyHtoD( reinterpret_cast<float3*>( mesh1.vertices ), v, mesh1.vertexCount );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh1;

		size_t			  geomTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTempTris1 ), geomTempSize );

		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geomTris1 ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTempTris1, 0, geomTris1 ) );
	}

	hiprtScene			 scene;
	hiprtDevicePtr		 sceneTemp;
	hiprtSceneBuildInput sceneInput;
	{
		hiprtInstance instTris0;
		instTris0.type	   = hiprtInstanceTypeGeometry;
		instTris0.geometry = geomTris0;

		hiprtInstance instTris1;
		instTris1.type	   = hiprtInstanceTypeGeometry;
		instTris1.geometry = geomTris1;

		hiprtInstance instances[] = { instTris0, instTris1 };

		sceneInput.instanceCount = 2;
		sceneInput.instanceMasks = nullptr;
		malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
		copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), instances, sceneInput.instanceCount );

		constexpr float Offset = 0.3f;
		hiprtFrameSRT	frames[5];
		frames[0].translation = { -0.25f, -Offset, 0.0f };
		frames[0].scale		  = { 1.0f, 1.0f, 1.0f };
		frames[0].rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[0].time		  = 0.0f;
		frames[1].translation = { 0.0f, -Offset, 0.0f };
		frames[1].scale		  = { 1.0f, 1.0f, 1.0f };
		frames[1].rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[1].time		  = 0.35f;
		frames[2].translation = { 0.25f, -Offset, 0.0f };
		frames[2].scale		  = { 1.0f, 1.0f, 1.0f };
		frames[2].rotation	  = { 0.0f, 0.0f, 1.0f, hiprt::Pi * 0.25f };
		frames[2].time		  = 1.0f;
		frames[3].translation = { 0.0f, Offset, 0.0f };
		frames[3].scale		  = { 1.0f, 1.0f, 1.0f };
		frames[3].rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[3].time		  = 0.0f;
		frames[4].translation = { 0.0f, Offset, 0.0f };
		frames[4].scale		  = { 1.0f, 1.0f, 1.0f };
		frames[4].rotation	  = { 0.0f, 0.0f, 1.0f, hiprt::Pi * 0.5f };
		frames[4].time		  = 1.0f;

		sceneInput.frameCount = 5;
		malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), sceneInput.frameCount );
		copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), frames, sceneInput.frameCount );

		hiprtTransformHeader headers[2];
		headers[0].frameIndex = 0;
		headers[0].frameCount = 3;
		headers[1].frameIndex = 3;
		headers[1].frameCount = 2;
		malloc( reinterpret_cast<hiprtTransformHeader*&>( sceneInput.instanceTransformHeaders ), sceneInput.instanceCount );
		copyHtoD(
			reinterpret_cast<hiprtTransformHeader*>( sceneInput.instanceTransformHeaders ), headers, sceneInput.instanceCount );

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

		checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
		checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );
	}

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "MotionBlurKernel", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &scene, &dst, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "MotionBlur.png", dst, "MotionBlur.png" );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( sceneInput.instanceTransformHeaders );
	free( mesh0.vertices );
	free( mesh0.triangleIndices );
	free( mesh1.vertices );
	free( mesh1.triangleIndices );
	free( geomTempTris0 );
	free( geomTempTris1 );
	free( sceneTemp );
	free( dst );
	checkHiprt( hiprtDestroyGeometry( ctxt, geomTris0 ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geomTris1 ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, MotionBlurMatrix )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtGeometry			   geomTris0;
	hiprtDevicePtr			   geomTempTris0;
	hiprtTriangleMeshPrimitive mesh0;
	{
		mesh0.triangleCount	 = 1;
		mesh0.triangleStride = sizeof( uint3 );
		malloc( reinterpret_cast<uint3*&>( mesh0.triangleIndices ), mesh0.triangleCount );
		std::vector<uint32_t> idx( 3 * mesh0.triangleCount );
		std::iota( idx.begin(), idx.end(), 0 );
		copyHtoD(
			reinterpret_cast<uint3*>( mesh0.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh0.triangleCount );

		mesh0.vertexCount  = 3;
		mesh0.vertexStride = sizeof( float3 );
		malloc( reinterpret_cast<float3*&>( mesh0.vertices ), mesh0.vertexCount );
		constexpr float Scale = 0.15f;
		float3			v[]	  = {
			   { Scale * sinf( 0.0f ), Scale * cosf( 0.0f ), 0.0f },
			   { Scale * sinf( hiprt::Pi * 2.0f / 3.0f ), Scale * cosf( hiprt::Pi * 2.0f / 3.0f ), 0.0f },
			   { Scale * sinf( hiprt::Pi * 4.0f / 3.0f ), Scale * cosf( hiprt::Pi * 4.0f / 3.0f ), 0.0f } };
		copyHtoD( reinterpret_cast<float3*>( mesh0.vertices ), v, mesh0.vertexCount );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh0;

		size_t			  geomTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTempTris0 ), geomTempSize );

		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geomTris0 ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTempTris0, 0, geomTris0 ) );
	}

	hiprtGeometry			   geomTris1;
	hiprtDevicePtr			   geomTempTris1;
	hiprtTriangleMeshPrimitive mesh1;
	{
		mesh1.triangleCount	 = 1;
		mesh1.triangleStride = sizeof( uint3 );
		malloc( reinterpret_cast<uint3*&>( mesh1.triangleIndices ), mesh1.triangleCount );
		std::vector<uint32_t> idx( 3 * mesh1.triangleCount );
		std::iota( idx.begin(), idx.end(), 0 );
		copyHtoD(
			reinterpret_cast<uint3*>( mesh1.triangleIndices ), reinterpret_cast<uint3*>( idx.data() ), mesh1.triangleCount );

		mesh1.vertexCount  = 3;
		mesh1.vertexStride = sizeof( float3 );
		malloc( reinterpret_cast<float3*&>( mesh1.vertices ), mesh1.vertexCount );
		constexpr float Scale = 0.15f;
		float3			v[]	  = {
			   { Scale * sinf( 0.0f ), Scale * cosf( 0.0f ), 0.0f },
			   { Scale * sinf( hiprt::Pi * 2.0f / 3.0f ), Scale * cosf( hiprt::Pi * 2.0f / 3.0f ), 0.0f },
			   { Scale * sinf( hiprt::Pi * 4.0f / 3.0f ), Scale * cosf( hiprt::Pi * 4.0f / 3.0f ), 0.0f } };
		copyHtoD( reinterpret_cast<float3*>( mesh1.vertices ), v, mesh1.vertexCount );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh1;

		size_t			  geomTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( geomTempTris1 ), geomTempSize );

		checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geomTris1 ) );
		checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTempTris1, 0, geomTris1 ) );
	}

	hiprtScene			 scene;
	hiprtDevicePtr		 sceneTemp;
	hiprtSceneBuildInput sceneInput;
	{
		hiprtInstance instTris0;
		instTris0.type	   = hiprtInstanceTypeGeometry;
		instTris0.geometry = geomTris0;

		hiprtInstance instTris1;
		instTris1.type	   = hiprtInstanceTypeGeometry;
		instTris1.geometry = geomTris1;

		hiprtInstance instances[] = { instTris0, instTris1 };

		sceneInput.instanceCount = 2;
		sceneInput.instanceMasks = nullptr;
		malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
		copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), instances, sceneInput.instanceCount );

		constexpr float	 Offset = 0.3f;
		hiprtFrameMatrix matrices[5];

		matrices[0]				 = hiprtFrameMatrix{};
		matrices[0].matrix[0][0] = 1.0f;
		matrices[0].matrix[1][1] = 1.0f;
		matrices[0].matrix[2][2] = 1.0f;
		matrices[0].matrix[0][3] = -0.25f;
		matrices[0].matrix[1][3] = -Offset;
		matrices[0].matrix[2][3] = 0.0f;
		matrices[0].time		 = 0.0f;

		matrices[1]				 = hiprtFrameMatrix{};
		matrices[1].matrix[0][0] = 1.0f;
		matrices[1].matrix[1][1] = 1.0f;
		matrices[1].matrix[2][2] = 1.0f;
		matrices[1].matrix[0][3] = 0.0f;
		matrices[1].matrix[1][3] = -Offset;
		matrices[1].matrix[2][3] = 0.0f;
		matrices[1].time		 = 0.35f;

		matrices[2]				 = hiprtFrameMatrix{};
		matrices[2].matrix[0][0] = cosf( hiprt::Pi * 0.25f );
		matrices[2].matrix[0][1] = -sinf( hiprt::Pi * 0.25f );
		matrices[2].matrix[1][0] = sinf( hiprt::Pi * 0.25f );
		matrices[2].matrix[1][1] = cosf( hiprt::Pi * 0.25f );
		matrices[2].matrix[2][2] = 1.0f;
		matrices[2].matrix[0][3] = 0.25f;
		matrices[2].matrix[1][3] = -Offset;
		matrices[2].matrix[2][3] = 0.0f;
		matrices[2].time		 = 1.0f;

		matrices[3]				 = hiprtFrameMatrix{};
		matrices[3].matrix[0][0] = 1.0f;
		matrices[3].matrix[1][1] = 1.0f;
		matrices[3].matrix[2][2] = 1.0f;
		matrices[3].matrix[0][3] = 0.0f;
		matrices[3].matrix[1][3] = Offset;
		matrices[3].matrix[2][3] = 0.0f;
		matrices[3].time		 = 0.0f;

		matrices[4]				 = hiprtFrameMatrix{};
		matrices[4].matrix[0][0] = 0.0f;
		matrices[4].matrix[0][1] = -1.0f;
		matrices[4].matrix[1][0] = 1.0f;
		matrices[4].matrix[1][1] = 0.0f;
		matrices[4].matrix[2][2] = 1.0f;
		matrices[4].matrix[0][3] = 0.0f;
		matrices[4].matrix[1][3] = Offset;
		matrices[4].matrix[2][3] = 0.0f;
		matrices[4].time		 = 1.0f;

		sceneInput.frameCount = 5;
		sceneInput.frameType  = hiprtFrameTypeMatrix;
		malloc( reinterpret_cast<hiprtFrameMatrix*&>( sceneInput.instanceFrames ), sceneInput.frameCount );
		copyHtoD( reinterpret_cast<hiprtFrameMatrix*>( sceneInput.instanceFrames ), matrices, sceneInput.frameCount );

		hiprtTransformHeader headers[2];
		headers[0].frameIndex = 0;
		headers[0].frameCount = 3;
		headers[1].frameIndex = 3;
		headers[1].frameCount = 2;
		malloc( reinterpret_cast<hiprtTransformHeader*&>( sceneInput.instanceTransformHeaders ), sceneInput.instanceCount );
		copyHtoD(
			reinterpret_cast<hiprtTransformHeader*>( sceneInput.instanceTransformHeaders ), headers, sceneInput.instanceCount );

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
		malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

		checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
		checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );
	}

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "MotionBlurKernel", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &scene, &dst, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "MotionBlurMatrix.png", dst, "MotionBlur.png" );

	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( sceneInput.instanceTransformHeaders );
	free( mesh0.vertices );
	free( mesh0.triangleIndices );
	free( mesh1.vertices );
	free( mesh1.triangleIndices );
	free( geomTempTris0 );
	free( geomTempTris1 );
	free( sceneTemp );
	free( dst );
	checkHiprt( hiprtDestroyGeometry( ctxt, geomTris0 ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geomTris1 ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, MotionBlurSlerp )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtAABBListPrimitive list;
	list.aabbCount	= 1;
	list.aabbStride = 6 * sizeof( float );
	malloc( reinterpret_cast<float3*&>( list.aabbs ), 2 * list.aabbCount );
	float3 b[] = { { -1.5f, -0.5f, -0.5f }, { -0.5f, 0.5f, 0.5f } };
	copyHtoD( reinterpret_cast<float3*>( list.aabbs ), b, 2 * list.aabbCount );

	const uint32_t geomType_SPHERE = 1;

	hiprtGeometryBuildInput geomInput;
	geomInput.type				 = hiprtPrimitiveTypeAABBList;
	geomInput.primitive.aabbList = list;
	geomInput.geomType			 = 1;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	float3 mn, mx;
	hiprtExportGeometryAabb( ctxt, geom, mn, mx );

	hiprtScene			 scene;
	hiprtDevicePtr		 sceneTemp;
	hiprtSceneBuildInput sceneInput;

	hiprtInstance instance;
	instance.type	  = hiprtInstanceTypeGeometry;
	instance.geometry = geom;

	sceneInput.instanceCount			= 1;
	sceneInput.instanceMasks			= nullptr;
	sceneInput.instanceTransformHeaders = nullptr;
	malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
	copyHtoD( reinterpret_cast<hiprtInstance*>( sceneInput.instances ), &instance, sceneInput.instanceCount );

	hiprtFrameMatrix matrix0{};
	matrix0.matrix[0][0] = 1.0f;
	matrix0.matrix[1][1] = 1.0f;
	matrix0.matrix[2][2] = 1.0f;
	matrix0.time		 = 0.0f;
	hiprtFrameMatrix matrix1{};
	matrix1.matrix[0][0]		= -1.0f;
	matrix1.matrix[1][1]		= -1.0f;
	matrix1.matrix[2][2]		= 1.0f;
	matrix1.time				= 1.0f;
	hiprtFrameMatrix matrices[] = { matrix0, matrix1 };

	sceneInput.frameCount = 2;
	sceneInput.frameType  = hiprtFrameTypeMatrix;
	malloc( reinterpret_cast<hiprtFrameMatrix*&>( sceneInput.instanceFrames ), sceneInput.frameCount );
	copyHtoD( reinterpret_cast<hiprtFrameMatrix*>( sceneInput.instanceFrames ), matrices, sceneInput.frameCount );

	hiprtTransformHeader header{ 0, sceneInput.frameCount };
	malloc( reinterpret_cast<hiprtTransformHeader*&>( sceneInput.instanceTransformHeaders ), 1 );
	copyHtoD( reinterpret_cast<hiprtTransformHeader*>( sceneInput.instanceTransformHeaders ), &header, 1 );

	size_t sceneTempSize;
	checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );

	checkHiprt( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	checkHiprt( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	hiprtFuncNameSet funcName_unused;
	hiprtFuncNameSet funcNameSet;
	funcNameSet.intersectFuncName = "intersectSphere";
	// use 'funcNameSet' at slot 'geomType_SPHERE'
	// for the previous slot, use empty functions.
	std::vector<hiprtFuncNameSet> funcNameSets = { funcName_unused, funcNameSet };
	cudaFunction_t func;
	buildTraceKernel(
			ctxt,
			getRootDir() / "test/kernels/HiprtTestKernel.h",
			"MotionBlurSlerpKernel",
			func,
			std::nullopt,
			funcNameSets,
			2,
			1 );

	float4* spheres;
	malloc( spheres, list.aabbCount );
	float4 h = { -1.0f, 0.0f, 0.0f, 0.5f };
	copyHtoD( spheres, &h, list.aabbCount );

	hiprtFuncDataSet funcDataSet;
	funcDataSet.intersectFuncData = spheres;

	hiprtFuncTable funcTable;
	checkHiprt( hiprtCreateFuncTable( ctxt, 2, 1, funcTable ) );
	checkHiprt( hiprtSetFuncTable( ctxt, funcTable, geomType_SPHERE, 0, funcDataSet ) );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &scene, &dst, &funcTable, &res };
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "MotionBlurSlerp.png", dst, "MotionBlurSlerp.png" );

	free( list.aabbs );
	free( geomTemp );
	free( sceneTemp );
	free( sceneInput.instances );
	free( sceneInput.instanceFrames );
	free( sceneInput.instanceTransformHeaders );
	free( spheres );
	free( dst );
	checkHiprt( hiprtDestroyFuncTable( ctxt, funcTable ) );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyScene( ctxt, scene ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, Rebuild )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 1;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 3, 4, 5 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 3;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );

	float3 v[] = {
		{ 0.15f, 0.40f, 0.0f },
		{ 0.35f, 0.40f, 0.0f },
		{ 0.25f, 0.60f, 0.0f },
		{ 0.65f, 0.40f, 0.0f },
		{ 0.85f, 0.40f, 0.0f },
		{ 0.75f, 0.60f, 0.0f } };

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "MeshIntersectionKernel", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &geom, &dst, &res };

	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "Rebuild0.png", dst, "Rebuild0.png" );

	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v + 3, mesh.vertexCount );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "Rebuild1.png", dst, "Rebuild1.png" );

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, Update )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 1;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 3, 4, 5 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 3;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );

	float3 v[] = {
		{ 0.15f, 0.40f, 0.0f },
		{ 0.35f, 0.40f, 0.0f },
		{ 0.25f, 0.60f, 0.0f },
		{ 0.65f, 0.40f, 0.0f },
		{ 0.85f, 0.40f, 0.0f },
		{ 0.75f, 0.60f, 0.0f } };

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "MeshIntersectionKernel", func );

	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	uint2 res = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };

	void* args[] = { &geom, &dst, &res };

	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "Update0.png", dst, "Update0.png" );

	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v + 3, mesh.vertexCount );
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationUpdate, geomInput, options, geomTemp, 0, geom ) );
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, args );
	validateAndWriteImage( "Update1.png", dst, "Update1.png" );

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	free( dst );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, PlocFallback )
{
	hiprtContext ctxt;
	hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 100000u;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	std::vector<uint3> idx( mesh.triangleCount );
	for ( uint3& i : idx )
		i = { 0, 1, 2 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), idx.data(), mesh.triangleCount );

	mesh.vertexCount  = 3;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	float3 v[] = { { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.5f, 1.0f, 0.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;
	checkHiprt( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	checkHiprt( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	checkHiprt( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::chrono::nanoseconds bvhBuildTime = end - begin;
	std::cout << "Ploc fallback build Time " << std::chrono::duration_cast<std::chrono::seconds>( bvhBuildTime ).count() << " s"
			  << std::endl;

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	checkHiprt( hiprtDestroyGeometry( ctxt, geom ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, TraceKernel )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	constexpr uint32_t sharedStackSize	  = 16;
	constexpr uint32_t blockWidth		  = 8;
	constexpr uint32_t blockHeight		  = 8;
	constexpr uint32_t blockSize		  = blockWidth * blockHeight;
	std::string		   blockSizeDef		  = "-DBLOCK_SIZE=" + std::to_string( blockSize );
	std::string		   sharedStackSizeDef = "-DSHARED_STACK_SIZE=" + std::to_string( sharedStackSize );

	std::vector<const char*> opts;
	opts.push_back( blockSizeDef.c_str() );
	opts.push_back( sharedStackSizeDef.c_str() );

	cudaFunction_t func;
	buildTraceKernel( ctxt, getRootDir() / "test/kernels/HiprtTestKernel.h", "TraceKernel", func, opts );
	int numRegs;
	// checkOro( cudaFuncGetAttribute( &numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, func ) );

	int numSmem;
	// checkOro( cudaFuncGetAttribute( &numSmem, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func ) );

	std::cout << "Trace kernel: registers " << numRegs << ", shared memory " << numSmem << std::endl;

	checkHiprt( hiprtDestroyContext( ctxt ) );
}

TEST_F( hiprtTest, BatchConstruction )
{
	hiprtContext ctxt;
	hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= 2;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	uint32_t idx[] = { 0, 1, 2, 3, 4, 5 };
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( idx ), mesh.triangleCount );

	mesh.vertexCount  = 6;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	float3 v[] = {
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f },
		{ 0.5f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f },
		{ 1.0f, 0.0f, 1.0f },
		{ 0.5f, 1.0f, 1.0f } };
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), v, mesh.vertexCount );

	constexpr size_t GeomCount = 1000000;

	std::vector<hiprtGeometryBuildInput> geomInputs( GeomCount );
	for ( hiprtGeometryBuildInput& geomInput : geomInputs )
	{
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh;
		geomInput.geomType				 = 0;
	}

	hiprtBuildOptions options;
	options.buildFlags			   = hiprtBuildFlagBitPreferFastBuild;
	options.batchBuildMaxPrimCount = 64u;

	hiprtDevicePtr tempGeomBuffer = nullptr;
	size_t		   tempGeomSize;
	checkHiprt( hiprtGetGeometriesBuildTemporaryBufferSize( ctxt, GeomCount, geomInputs.data(), options, tempGeomSize ) );
	malloc( reinterpret_cast<uint8_t*&>( tempGeomBuffer ), tempGeomSize );

	std::vector<hiprtGeometry>	geometries( GeomCount );
	std::vector<hiprtGeometry*> geomAddrs( GeomCount );
	for ( size_t i = 0; i < GeomCount; ++i )
		geomAddrs[i] = &geometries[i];
	checkHiprt( hiprtCreateGeometries( ctxt, GeomCount, geomInputs.data(), options, geomAddrs.data() ) );

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	checkHiprt( hiprtBuildGeometries(
		ctxt, hiprtBuildOperationBuild, GeomCount, geomInputs.data(), options, tempGeomBuffer, 0, geometries.data() ) );
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::chrono::nanoseconds bvhBuildTime = end - begin;
	std::cout << "Bvh batch build time " << std::chrono::duration_cast<std::chrono::seconds>( bvhBuildTime ).count() << " s"
			  << std::endl;

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( tempGeomBuffer );
	checkHiprt( hiprtDestroyGeometries( ctxt, GeomCount, geometries.data() ) );
}

int main( int argc, const char* argv[] )
{
	using namespace argparse;
	::testing::InitGoogleTest( &argc, const_cast<char**>( argv ) );

	CmdArguments   parsedArgs;
	ArgumentParser parser( "HIPRT", "hiprt tests" );
	parser.add_argument().names( { "-w", "--width" } ).description( "width" ).required( false );
	parser.add_argument().names( { "-h", "--height" } ).description( "height" ).required( false );
	parser.add_argument().names( { "-r", "--referencePath" } ).description( "path for reference images" ).required( false );
	parser.add_argument().names( { "-d", "--device" } ).description( "device" ).required( false );
	parser.parse( argc, argv );
	parser.print_help();

	if ( parser.exists( "w" ) )
	{
		parsedArgs.m_ww = parser.get<uint32_t>( "w" );
	}
	if ( parser.exists( "h" ) )
	{
		parsedArgs.m_ww = parser.get<uint32_t>( "h" );
	}
	if ( parser.exists( "r" ) )
	{
		parsedArgs.m_referencePath = parser.get<std::string>( "r" ).c_str();
	}
	if ( parser.exists( "d" ) )
	{
		parsedArgs.m_deviceIdx = parser.get<uint32_t>( "d" );
	}

	::testing::AddGlobalTestEnvironment( new InitCommandlineArgs( parsedArgs ) );
	int ret = RUN_ALL_TESTS();
	return ret;
}
