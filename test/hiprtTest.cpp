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
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <nvrtc.h>
#include <test/hiprtTest.h>
#include <test/CornellBox.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <contrib/stbi/stbi_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <contrib/stbi/stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"
#include "common/allocator.h"
#include "common/bvhbuilder.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

CmdArguments g_parsedArgs;

void checkOro( cudaError res, const source_location& location )
{
	if ( res != cudaSuccess )
	{
		// const char* msg;
		cudaGetErrorString( res );
		std::cerr << "Orochi error: '" << res << "' on line " << location.line() << " "
				  << " in '" << location.file_name() << "'." << std::endl;
		std::abort();
	}
}
void checkOro( CUresult res, const source_location& location )
{
	if ( res != CUDA_SUCCESS )
	{
		const char* msg;
		cuGetErrorString( res, &msg );
		std::cerr << "Orochi error: '" << msg << "' on line " << location.line() << " "
				  << " in '" << location.file_name() << "'." << std::endl;
		std::abort();
	}
}
void checkOrortc( nvrtcResult res, const source_location& location )
{
	if ( res != NVRTC_SUCCESS )
	{
		std::cerr << "Orortc error: '" << nvrtcGetErrorString( res ) << "' [ " << res << " ] on line " << location.line()
				  << " "
				  << " in '" << location.file_name() << "'." << std::endl;
		std::abort();
	}
}

void checkHiprt( hiprtError res, const source_location& location )
{
	if ( res != hiprtSuccess )
	{
		std::cerr << "Hiprt error: '" << res << "' on line " << location.line() << " "
				  << " in '" << location.file_name() << "'." << std::endl;
		std::abort();
	}
}

namespace
{
std::string getNvrtcCompiledBinaryForTest( nvrtcProgram prog )
{
	size_t ptxSize = 0;
	checkOrortc( nvrtcGetPTXSize( prog, &ptxSize ) );
	if ( ptxSize > 0 )
	{
		std::string ptx( ptxSize, '\0' );
		checkOrortc( nvrtcGetPTX( prog, ptx.data() ) );
		return ptx;
	}

	size_t cubinSize = 0;
	checkOrortc( nvrtcGetCUBINSize( prog, &cubinSize ) );
	if ( cubinSize > 0 )
	{
		std::string cubin( cubinSize, '\0' );
		checkOrortc( nvrtcGetCUBIN( prog, cubin.data() ) );
		return cubin;
	}

	return {};
}

bool isFatbinBinaryForTest( const std::string_view binary )
{
	constexpr std::string_view magic = "__CLANG_OFFLOAD_";
	return binary.size() >= magic.size() && binary.substr( 0, magic.size() ) == magic;
}

std::string quoteShellArgForTest( const std::string& arg )
{
	std::string escaped = "'";
	for ( const char ch : arg )
	{
		if ( ch == '\'' ) escaped += "'\\''";
		else escaped += ch;
	}
	escaped += "'";
	return escaped;
}

std::string findCudaCompilerForTest()
{
	const std::string preferredExternal = getEnvVariable( "HIPRT_EXTERNAL_DEVICE_COMPILER" );
	const bool		preferMxcc		  = preferredExternal == "mxcc";
	const bool		preferCucc		  = preferredExternal == "cucc";
	const bool		preferNvcc		  = preferredExternal == "nvcc";

	const std::vector<std::string> candidates = {
		preferMxcc ? "/opt/maca/mxgpu_llvm/bin/mxcc" : std::string(),
		preferCucc ? "/opt/maca/tools/cu-bridge/bin/cucc" : std::string(),
		preferNvcc ? "nvcc" : std::string(),
		getEnvVariable( "HIPRT_CUDA_COMPILER" ),
		getEnvVariable( "CUDACXX" ),
		getEnvVariable( "CUBIN_COMPILER" ),
		"/opt/maca/mxgpu_llvm/bin/mxcc",
		getEnvVariable( "CUDA_PATH" ).empty() ? std::string() : getEnvVariable( "CUDA_PATH" ) + "/bin/nvcc",
		"/root/cu-bridge/CUDA_DIR/bin/nvcc",
		"/opt/maca/tools/cu-bridge/bin/cucc",
		"/opt/maca/tools/cu-bridge/CUDA_DIR/bin/nvcc",
		"nvcc",
	};

	for ( const auto& candidate : candidates )
	{
		if ( candidate.empty() ) continue;
		if ( candidate == "nvcc" ) return candidate;
		if ( std::filesystem::exists( candidate ) ) return candidate;
	}

	return {};
}

std::string compileSourceToCubinForTest(
	const std::filesystem::path& srcPath, const std::vector<const char*>& options, const std::filesystem::path& rootDir )
{
	const std::string compiler = findCudaCompilerForTest();
	if ( compiler.empty() ) return {};

	const std::filesystem::path absoluteSrcPath  = std::filesystem::absolute( srcPath );
	const std::filesystem::path absoluteRootPath = std::filesystem::absolute( rootDir );
	const std::filesystem::path tempDir =
		std::filesystem::temp_directory_path() /
		( "hiprt-test-bitcode-" + std::to_string( std::chrono::steady_clock::now().time_since_epoch().count() ) );
	std::filesystem::create_directories( tempDir );

	const bool useMxcc = std::filesystem::path( compiler ).filename() == "mxcc";
	const std::filesystem::path cubinPath = tempDir / ( useMxcc ? "trace_kernel.fatbin" : "trace_kernel.cubin" );

	std::ostringstream cmd;
	if ( useMxcc )
	{
		const std::string macaPath = getEnvVariable( "MACA_PATH" ).empty() ? "/opt/maca" : getEnvVariable( "MACA_PATH" );
		const std::string cudaPath =
			getEnvVariable( "CUDA_PATH" ).empty() ? macaPath + "/tools/cu-bridge" : getEnvVariable( "CUDA_PATH" );
		const std::string offloadArch =
			getEnvVariable( "MXCC_OFFLOAD_ARCH" ).empty() ? "xcore1000" : getEnvVariable( "MXCC_OFFLOAD_ARCH" );

		cmd << quoteShellArgForTest( compiler ) << " -O3 -std=c++17 -fatbin -use-fast-math"
			<< " -x maca -fgpu-rdc --include cuda_runtime.h -D__CUDACC__"
			<< " -I" << quoteShellArgForTest( absoluteRootPath.string() )
			<< " -I" << quoteShellArgForTest( ( absoluteRootPath / "test" ).string() )
			<< " -I" << quoteShellArgForTest( ( absoluteRootPath / "contrib/Orochi" ).string() )
			<< " -I" << quoteShellArgForTest( cudaPath + "/include" )
			<< " -I" << quoteShellArgForTest( macaPath + "/include" )
			<< " --offload-arch=" << offloadArch
			<< " " << quoteShellArgForTest( absoluteSrcPath.string() );
	}
	else
	{
		cmd << quoteShellArgForTest( compiler ) << " -x cu " << quoteShellArgForTest( absoluteSrcPath.string() )
			<< " -O3 -std=c++17 --device-c -cubin --use_fast_math"
			<< " -I" << quoteShellArgForTest( absoluteRootPath.string() )
			<< " -I" << quoteShellArgForTest( ( absoluteRootPath / "test" ).string() )
			<< " -I" << quoteShellArgForTest( ( absoluteRootPath / "contrib/Orochi" ).string() );
	}
	for ( const char* option : options )
		cmd << " " << option;
	cmd << " -o " << quoteShellArgForTest( cubinPath.string() );

	if ( std::system( cmd.str().c_str() ) != 0 ) return {};

	std::ifstream file( cubinPath, std::ios::in | std::ios::binary | std::ios::ate );
	if ( !file.is_open() ) return {};

	const size_t size = static_cast<size_t>( file.tellg() );
	file.seekg( 0, std::ios::beg );
	std::string cubin( size, '\0' );
	file.read( cubin.data(), static_cast<std::streamsize>( size ) );
	return cubin;
}
} // namespace

std::string getEnvVariable( const std::string& key )
{
#if defined( __WINDOWS__ )
	char*  buffer	   = nullptr;
	size_t bufferCount = 0;
	_dupenv_s( &buffer, &bufferCount, key.c_str() );
	const std::string val = buffer != nullptr ? buffer : "";
	delete[] buffer;
#else
	const char* const env = getenv( key.c_str() );
	const std::string val = ( env == nullptr ) ? std::string() : std::string( env );
#endif
	return val;
}

std::filesystem::path getRootDir()
{
	std::string val = getEnvVariable( "HIPRT_PATH" );
	if ( val.empty() ) val = "..";
	return std::filesystem::path( val );
}

inline float3 operator-( float3& a, float3& b ) { return float3{ a.x - b.x, a.y - b.y, a.z - b.z }; }

struct EmbreeBuildNode
{
	float3	 m_childAabbsMin[2];
	float3	 m_childAabbsMax[2];
	uint32_t m_childIndices[2];
};

struct BuilderContext
{
	PoolAllocator<EmbreeBuildNode, 16> m_nodeAllocator;
	PoolAllocator<uint32_t, 16>		   m_leafAllocator;
	void*							   m_geomData;
};

struct GeometryData
{
	const float3*	m_vertices;
	const uint32_t* m_indices;
	const uint2*	m_pairIndices = nullptr;
};

void hiprtTest::SetUp()
{
	checkOro( cuInit( 0 ) );
	cudaError deviceGetError = cudaGetDevice( &m_cudaDevice);
	if ( deviceGetError != cudaSuccess )
	{
		// if failed, try to understand what happened.
		int deviceCountCuda = 0;
		cudaGetDeviceCount( &deviceCountCuda );

		std::cout << "ERROR detected inside cudaGetDevice." << std::endl;
		std::cout << "number of CUDA devices detected = " << deviceCountCuda << std::endl;
		if ( deviceCountCuda == 0 )
			std::cout << "NO COMPATIBLE DEVICE FOUND. check your driver." << std::endl;

		checkOro( deviceGetError );
	}

	cudaDeviceProp props;
	checkOro( cudaGetDeviceProperties( &props, m_cudaDevice ) );
	std::cout << "Executing on '" << props.name << "'" << std::endl;
	std::cout << "pciDeviceID on '" << props.pciDeviceID << "'" << std::endl;
	std::cout << "m_cudaDevice on '" << m_cudaDevice << "'" << std::endl;

	if ( std::string( props.name ).find( "NVIDIA" ) != std::string::npos )
		m_ctxtInput.deviceType = hiprtDeviceNVIDIA;
	m_ctxtInput.device = cudaGetDevice( &m_cudaDevice );
}

void hiprtTest::buildBvh( hiprtGeometryBuildInput& buildInput )
{
	std::vector<hiprtInternalNode> internalNodes;
	std::vector<Aabb>			   primBoxes;
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh )
	{
		primBoxes.resize( buildInput.primitive.triangleMesh.triangleCount );
		std::vector<uint8_t> verticesRaw(
			buildInput.primitive.triangleMesh.vertexCount * buildInput.primitive.triangleMesh.vertexStride );
		std::vector<uint8_t> trianglesRaw(
			buildInput.primitive.triangleMesh.triangleCount * buildInput.primitive.triangleMesh.triangleStride );
		copyDtoH(
			verticesRaw.data(),
			reinterpret_cast<uint8_t*>( buildInput.primitive.triangleMesh.vertices ),
			buildInput.primitive.triangleMesh.vertexCount * buildInput.primitive.triangleMesh.vertexStride );
		copyDtoH(
			trianglesRaw.data(),
			reinterpret_cast<uint8_t*>( buildInput.primitive.triangleMesh.triangleIndices ),
			buildInput.primitive.triangleMesh.triangleCount * buildInput.primitive.triangleMesh.triangleStride );
		for ( uint32_t i = 0; i < buildInput.primitive.triangleMesh.triangleCount; ++i )
		{
			uint3 triangle =
				*reinterpret_cast<uint3*>( trianglesRaw.data() + i * buildInput.primitive.triangleMesh.triangleStride );
			float3 v0 = *reinterpret_cast<const float3*>(
				verticesRaw.data() + triangle.x * buildInput.primitive.triangleMesh.vertexStride );
			float3 v1 = *reinterpret_cast<const float3*>(
				verticesRaw.data() + triangle.y * buildInput.primitive.triangleMesh.vertexStride );
			float3 v2 = *reinterpret_cast<const float3*>(
				verticesRaw.data() + triangle.z * buildInput.primitive.triangleMesh.vertexStride );
			primBoxes[i].reset();
			primBoxes[i].grow( v0 );
			primBoxes[i].grow( v1 );
			primBoxes[i].grow( v2 );
		}
		BvhBuilder::build( buildInput.primitive.triangleMesh.triangleCount, primBoxes, internalNodes );
	}
	else if ( buildInput.type == hiprtPrimitiveTypeAABBList )
	{
		primBoxes.resize( buildInput.primitive.aabbList.aabbCount );
		std::vector<uint8_t> primBoxesRaw( buildInput.primitive.aabbList.aabbCount * buildInput.primitive.aabbList.aabbStride );
		copyDtoH(
			primBoxesRaw.data(),
			reinterpret_cast<uint8_t*>( buildInput.primitive.aabbList.aabbs ),
			buildInput.primitive.aabbList.aabbCount * buildInput.primitive.aabbList.aabbStride );
		for ( uint32_t i = 0; i < buildInput.primitive.aabbList.aabbCount; ++i )
		{
			float4* ptr = reinterpret_cast<float4*>( primBoxesRaw.data() + i * buildInput.primitive.aabbList.aabbStride );
			primBoxes[i].m_min = hiprt::make_float3( ptr[0] );
			primBoxes[i].m_max = hiprt::make_float3( ptr[1] );
		}
		BvhBuilder::build( buildInput.primitive.aabbList.aabbCount, primBoxes, internalNodes );
	}

	std::vector<hiprtLeafNode> leafNodes( primBoxes.size() );
	for ( uint32_t i = 0; i < primBoxes.size(); ++i )
	{
		leafNodes[i].primID	 = i;
		leafNodes[i].aabbMin = primBoxes[i].m_min;
		leafNodes[i].aabbMax = primBoxes[i].m_max;
	}

	buildInput.nodeList.nodeCount = static_cast<uint32_t>( leafNodes.size() );

	malloc( reinterpret_cast<hiprtLeafNode*&>( buildInput.nodeList.leafNodes ), leafNodes.size() );
	copyHtoD( reinterpret_cast<hiprtLeafNode*>( buildInput.nodeList.leafNodes ), leafNodes.data(), leafNodes.size() );

	malloc( reinterpret_cast<hiprtInternalNode*&>( buildInput.nodeList.internalNodes ), internalNodes.size() );
	copyHtoD(
		reinterpret_cast<hiprtInternalNode*>( buildInput.nodeList.internalNodes ), internalNodes.data(), internalNodes.size() );
}

void hiprtTest::buildEmbreeBvh(
	RTCDevice embreeDevice, std::vector<RTCBuildPrimitive>& embreePrims, void* geomData, hiprtBvhNodeList& nodeList )

{
	const float Alpha = 1.5f;
	enum
	{
		LeafFlag = 1 << 30
	};

	size_t primCount	= embreePrims.size();
	size_t primCapacity = Alpha * embreePrims.size();
	embreePrims.resize( primCapacity );

	BuilderContext context;
	context.m_geomData = geomData;

	RTCBVH			  embreeBvh		  = rtcNewBVH( embreeDevice );
	RTCBuildArguments embreeArgs	  = rtcDefaultBuildArguments();
	embreeArgs.byteSize				  = sizeof( embreeArgs );
	embreeArgs.buildQuality			  = RTC_BUILD_QUALITY_HIGH;
	embreeArgs.maxBranchingFactor	  = 2;
	embreeArgs.bvh					  = embreeBvh;
	embreeArgs.primitives			  = embreePrims.data();
	embreeArgs.primitiveCount		  = primCount;
	embreeArgs.primitiveArrayCapacity = primCapacity;
	embreeArgs.minLeafSize			  = 1;
	embreeArgs.maxLeafSize			  = 1;
	embreeArgs.splitPrimitive		  = nullptr;
	embreeArgs.userPtr				  = &context;

	embreeArgs.createNode =
		[]( [[maybe_unused]] RTCThreadLocalAllocator allocator, [[maybe_unused]] uint32_t childCount, void* userPtr ) -> void* {
		BuilderContext*	 ctxt = reinterpret_cast<BuilderContext*>( userPtr );
		uint32_t		 handle;
		EmbreeBuildNode* ptr;
		ctxt->m_nodeAllocator.allocate( &handle, &ptr );
		return reinterpret_cast<void*>( static_cast<uintptr_t>( handle ) );
	};

	embreeArgs.setNodeChildren = []( void* nodePtr, void** children, uint32_t childCount, void* userPtr ) {
		BuilderContext*	 ctxt = reinterpret_cast<BuilderContext*>( userPtr );
		EmbreeBuildNode* node = ctxt->m_nodeAllocator.item( static_cast<uint32_t>( reinterpret_cast<uintptr_t>( nodePtr ) ) );
		for ( uint32_t i = 0; i < childCount; i++ )
		{
			node->m_childIndices[i] = static_cast<uint32_t>( reinterpret_cast<uintptr_t>( children[i] ) );
		}
	};

	embreeArgs.setNodeBounds = []( void* nodePtr, const struct RTCBounds** bounds, uint32_t childCount, void* userPtr ) {
		BuilderContext*	 ctxt = reinterpret_cast<BuilderContext*>( userPtr );
		EmbreeBuildNode* node = ctxt->m_nodeAllocator.item( static_cast<uint32_t>( reinterpret_cast<uintptr_t>( nodePtr ) ) );
		for ( uint32_t i = 0; i < childCount; i++ )
		{
			node->m_childAabbsMin[i] = float3{ bounds[i]->lower_x, bounds[i]->lower_y, bounds[i]->lower_z };
			node->m_childAabbsMax[i] = float3{ bounds[i]->upper_x, bounds[i]->upper_y, bounds[i]->upper_z };
		}
	};

	embreeArgs.createLeaf = []( [[maybe_unused]] RTCThreadLocalAllocator allocator,
								const struct RTCBuildPrimitive*			 primitives,
								[[maybe_unused]] size_t					 primitiveCount,
								void*									 userPtr ) -> void* {
		BuilderContext* ctxt = reinterpret_cast<BuilderContext*>( userPtr );
		uint32_t		handle;
		uint32_t*		ptr;
		ctxt->m_leafAllocator.allocate( &handle, &ptr );
		*ptr = primitives->primID;
		return reinterpret_cast<void*>( static_cast<uintptr_t>( handle | LeafFlag ) );
	};

	if ( geomData == nullptr )
	{
		embreeArgs.splitPrimitive = []( const struct RTCBuildPrimitive* primitive,
										uint32_t						dimension,
										float							position,
										struct RTCBounds*				leftBounds,
										struct RTCBounds*				rightBounds,
										[[maybe_unused]] void*			userPtr ) {
			leftBounds->lower_x = rightBounds->lower_x = primitive->lower_x;
			leftBounds->lower_y = rightBounds->lower_y = primitive->lower_y;
			leftBounds->lower_z = rightBounds->lower_z = primitive->lower_z;
			leftBounds->upper_x = rightBounds->upper_x = primitive->upper_x;
			leftBounds->upper_y = rightBounds->upper_y = primitive->upper_y;
			leftBounds->upper_z = rightBounds->upper_z = primitive->upper_z;
			( &leftBounds->upper_x )[dimension]		   = position;
			( &rightBounds->lower_x )[dimension]	   = position;
		};
	}
	else
	{
		embreeArgs.splitPrimitive = []( const struct RTCBuildPrimitive* primitive,
										uint32_t						dimension,
										float							position,
										struct RTCBounds*				leftBounds,
										struct RTCBounds*				rightBounds,
										void*							userPtr ) {
			BuilderContext* ctxt	 = reinterpret_cast<BuilderContext*>( userPtr );
			GeometryData*	geomData = reinterpret_cast<GeometryData*>( ctxt->m_geomData );

			auto splitTriangle =
				[]( float3( &vertices )[3], uint32_t axis, float position, const Aabb& box, Aabb& leftBox, Aabb& rightBox ) {
					const float3* v1 = &vertices[2];
					for ( uint32_t i = 0; i < 3; i++ )
					{
						const float3* v0 = v1;
						v1				 = &vertices[i];
						float v0p		 = ( &v0->x )[axis];
						float v1p		 = ( &v1->x )[axis];

						if ( v0p <= position ) leftBox.grow( *v0 );
						if ( v0p >= position ) rightBox.grow( *v0 );

						if ( ( v0p < position && v1p > position ) || ( v0p > position && v1p < position ) )
						{
							float3 t = hiprt::mix( *v0, *v1, fmaxf( fminf( ( position - v0p ) / ( v1p - v0p ), 1.0f ), 0.0f ) );
							leftBox.grow( t );
							rightBox.grow( t );
						}
					}

					( &leftBox.m_max.x )[axis]	= position;
					( &rightBox.m_min.x )[axis] = position;
					leftBox.intersect( box );
					rightBox.intersect( box );
				};

			uint2 primID = hiprt::make_uint2( primitive->primID );
			if ( geomData->m_pairIndices != nullptr ) primID = geomData->m_pairIndices[primitive->primID];

			Aabb box, leftBox, rightBox;
			box.grow( float3{ primitive->lower_x, primitive->lower_y, primitive->lower_z } );
			box.grow( float3{ primitive->upper_x, primitive->upper_y, primitive->upper_z } );

			float3 vertices[3];
			vertices[0] = geomData->m_vertices[geomData->m_indices[3 * primID.x + 0]];
			vertices[1] = geomData->m_vertices[geomData->m_indices[3 * primID.x + 1]];
			vertices[2] = geomData->m_vertices[geomData->m_indices[3 * primID.x + 2]];
			splitTriangle( vertices, dimension, position, box, leftBox, rightBox );

			if ( primID.x != primID.y )
			{
				Aabb secLeftBox, secRightBox;
				vertices[0] = geomData->m_vertices[geomData->m_indices[3 * primID.y + 0]];
				vertices[1] = geomData->m_vertices[geomData->m_indices[3 * primID.y + 1]];
				vertices[2] = geomData->m_vertices[geomData->m_indices[3 * primID.y + 2]];
				splitTriangle( vertices, dimension, position, box, secLeftBox, secRightBox );
				leftBox.grow( secLeftBox );
				rightBox.grow( secRightBox );
			}

			leftBounds->lower_x = leftBox.m_min.x;
			leftBounds->lower_y = leftBox.m_min.y;
			leftBounds->lower_z = leftBox.m_min.z;
			leftBounds->upper_x = leftBox.m_max.x;
			leftBounds->upper_y = leftBox.m_max.y;
			leftBounds->upper_z = leftBox.m_max.z;

			rightBounds->lower_x = rightBox.m_min.x;
			rightBounds->lower_y = rightBox.m_min.y;
			rightBounds->lower_z = rightBox.m_min.z;
			rightBounds->upper_x = rightBox.m_max.x;
			rightBounds->upper_y = rightBox.m_max.y;
			rightBounds->upper_z = rightBox.m_max.z;
		};
	}

	rtcBuildBVH( &embreeArgs );

	const uint32_t nodeCount = context.m_nodeAllocator.count();
	const uint32_t leafCount = context.m_leafAllocator.count();
	uint32_t	   leafIndex = 0;

	std::vector<hiprtInternalNode> internalNodes( nodeCount );
	std::vector<hiprtLeafNode>	   leafNodes( leafCount );
	for ( uint32_t i = 0; i < nodeCount; ++i )
	{
		EmbreeBuildNode* embreeNode = context.m_nodeAllocator.item( i );

		hiprtInternalNode internalNode{};
		internalNode.aabbMin = hiprt::make_float3( hiprt::FltMax );
		internalNode.aabbMax = hiprt::make_float3( -hiprt::FltMax );

		for ( uint32_t j = 0; j < 2; j++ )
		{
			internalNode.aabbMin = hiprt::min( internalNode.aabbMin, embreeNode->m_childAabbsMin[j] );
			internalNode.aabbMax = hiprt::max( internalNode.aabbMax, embreeNode->m_childAabbsMax[j] );

			uint32_t		 childIndex = embreeNode->m_childIndices[j];
			hiprtBvhNodeType childType	= hiprtBvhNodeTypeInternal;
			if ( childIndex & LeafFlag )
			{
				uint32_t* primID = context.m_leafAllocator.item( childIndex & ( ~LeafFlag ) );
				childIndex		 = leafIndex;
				childType		 = hiprtBvhNodeTypeLeaf;

				hiprtLeafNode leafNode{};
				leafNode.aabbMin = embreeNode->m_childAabbsMin[j];
				leafNode.aabbMax = embreeNode->m_childAabbsMax[j];
				leafNode.primID	 = *primID;

				leafNodes[leafIndex++] = leafNode;
			}

			internalNode.childIndices[j]   = childIndex;
			internalNode.childNodeTypes[j] = childType;
		}

		internalNodes[i] = internalNode;
	}
	assert( leafIndex == leafCount );

	rtcReleaseBVH( embreeBvh );

	nodeList.nodeCount = static_cast<uint32_t>( leafNodes.size() );

	malloc( reinterpret_cast<hiprtLeafNode*&>( nodeList.leafNodes ), leafNodes.size() );
	copyHtoD( reinterpret_cast<hiprtLeafNode*>( nodeList.leafNodes ), leafNodes.data(), leafNodes.size() );

	malloc( reinterpret_cast<hiprtInternalNode*&>( nodeList.internalNodes ), internalNodes.size() );
	copyHtoD( reinterpret_cast<hiprtInternalNode*>( nodeList.internalNodes ), internalNodes.data(), internalNodes.size() );
}

void hiprtTest::buildEmbreeGeometryBvh(
	RTCDevice embreeDevice, const float3* vertices, const uint32_t* indices, hiprtGeometryBuildInput& buildInput )
{
	uint32_t triangleCount = buildInput.primitive.triangleMesh.triangleCount;

	GeometryData geomData;
	geomData.m_vertices = vertices;
	geomData.m_indices	= indices;

	std::vector<RTCBuildPrimitive> embreePrims;

	if ( triangleCount > 2 )
	{
		auto tryPairTriangles = [&]( const uint3& a, const uint3& b ) {
			uint3 lb = hiprt::make_uint3( 3 );

			lb.x = ( b.x == a.x ) ? 0 : lb.x;
			lb.y = ( b.y == a.x ) ? 0 : lb.y;
			lb.z = ( b.z == a.x ) ? 0 : lb.z;

			lb.x = ( b.x == a.y ) ? 1 : lb.x;
			lb.y = ( b.y == a.y ) ? 1 : lb.y;
			lb.z = ( b.z == a.y ) ? 1 : lb.z;

			lb.x = ( b.x == a.z ) ? 2 : lb.x;
			lb.y = ( b.y == a.z ) ? 2 : lb.y;
			lb.z = ( b.z == a.z ) ? 2 : lb.z;

			if ( ( lb.x == 3 ) + ( lb.y == 3 ) + ( lb.z == 3 ) <= 1 ) return lb;
			return hiprt::make_uint3( hiprt::InvalidValue );
		};

		std::vector<uint2> pairIndices;
		uint32_t		   groups = hiprt::DivideRoundUp( triangleCount, 32 );
		for ( uint32_t i = 0; i < groups; ++i )
		{
			const uint32_t	  offset = i * 32;
			std::vector<bool> active( 32 );
			for ( uint32_t j = 0; j < 32; ++j )
				active[j] = offset + j < triangleCount;

			for ( uint32_t j = 0; j < 32; ++j )
			{
				if ( !active[j] ) continue;
				uint2 pair		 = hiprt::make_uint2( offset + j );
				uint3 triIndices = *reinterpret_cast<const uint3*>( &indices[3 * ( offset + j )] );
				for ( uint32_t k = j + 1; k < 32; ++k )
				{
					if ( !active[k] ) continue;
					uint3 secondTriIndices = *reinterpret_cast<const uint3*>( &indices[3 * ( offset + k )] );
					bool  pairable		   = tryPairTriangles( secondTriIndices, triIndices ).x != hiprt::InvalidValue;
					if ( pairable )
					{
						pair.y	  = offset + k;
						active[k] = false;
						break;
					}
				}
				pairIndices.push_back( pair );
				active[j] = false;
			}
		}

		buildInput.primitive.triangleMesh.trianglePairCount = static_cast<uint32_t>( pairIndices.size() );
		malloc( reinterpret_cast<uint2*&>( buildInput.primitive.triangleMesh.trianglePairIndices ), pairIndices.size() );
		copyHtoD(
			reinterpret_cast<uint2*>( buildInput.primitive.triangleMesh.trianglePairIndices ),
			pairIndices.data(),
			pairIndices.size() );

		embreePrims.resize( pairIndices.size() );
		for ( size_t i = 0; i < pairIndices.size(); ++i )
		{
			Aabb box;
			for ( uint32_t j = 0; j < 2; ++j )
			{
				uint3 triIndices = *reinterpret_cast<const uint3*>( &indices[3 * ( &pairIndices[i].x )[j]] );
				box.grow( vertices[triIndices.x] );
				box.grow( vertices[triIndices.y] );
				box.grow( vertices[triIndices.z] );

				embreePrims[i].primID  = static_cast<uint32_t>( i );
				embreePrims[i].lower_x = box.m_min.x;
				embreePrims[i].lower_y = box.m_min.y;
				embreePrims[i].lower_z = box.m_min.z;
				embreePrims[i].upper_x = box.m_max.x;
				embreePrims[i].upper_y = box.m_max.y;
				embreePrims[i].upper_z = box.m_max.z;
			}
		}

		geomData.m_pairIndices = pairIndices.data();

		if ( embreePrims.size() > 1 ) buildEmbreeBvh( embreeDevice, embreePrims, &geomData, buildInput.nodeList );
	}
	else
	{
		embreePrims.resize( triangleCount );
		for ( uint32_t i = 0; i < triangleCount; ++i )
		{
			Aabb box;
			box.grow( vertices[indices[3 * i + 0]] );
			box.grow( vertices[indices[3 * i + 1]] );
			box.grow( vertices[indices[3 * i + 2]] );

			embreePrims[i].primID  = i;
			embreePrims[i].lower_x = box.m_min.x;
			embreePrims[i].lower_y = box.m_min.y;
			embreePrims[i].lower_z = box.m_min.z;
			embreePrims[i].upper_x = box.m_max.x;
			embreePrims[i].upper_y = box.m_max.y;
			embreePrims[i].upper_z = box.m_max.z;
		}

		if ( embreePrims.size() > 1 )
			buildEmbreeBvh( embreeDevice, embreePrims, &geomData, buildInput.nodeList );
		else
			buildInput.nodeList.nodeCount = 1;
	}
}

void hiprtTest::buildEmbreeSceneBvh(
	RTCDevice						  embreeDevice,
	const std::vector<Aabb>&		  geomBoxes,
	const std::vector<hiprtFrameSRT>& frames,
	hiprtSceneBuildInput&			  buildInput )
{
	uint32_t instanceCount = buildInput.instanceCount;

	struct BuilderContext
	{
		PoolAllocator<hiprtInternalNode, 16> nodeAllocator;
		PoolAllocator<uint32_t, 16>			 leafAllocator;
	} context;

	std::vector<RTCBuildPrimitive> embreePrims( instanceCount );
	for ( uint32_t i = 0; i < instanceCount; ++i )
	{
		const Aabb& geomBox = geomBoxes[i];
		Aabb		box		= geomBox;

		if ( !frames.empty() )
		{
			const hiprtFrameSRT& f = frames[i];

			float3 p[8];
			p[0] = geomBox.m_min;
			p[1] = float3{ geomBox.m_min.x, geomBox.m_min.y, geomBox.m_max.z };
			p[2] = float3{ geomBox.m_min.x, geomBox.m_max.y, geomBox.m_min.z };
			p[3] = float3{ geomBox.m_min.x, geomBox.m_max.y, geomBox.m_max.z };
			p[4] = float3{ geomBox.m_max.x, geomBox.m_min.y, geomBox.m_max.z };
			p[5] = float3{ geomBox.m_max.x, geomBox.m_max.y, geomBox.m_min.z };
			p[6] = float3{ geomBox.m_max.x, geomBox.m_max.y, geomBox.m_max.z };
			p[7] = geomBox.m_max;

			box.reset();
			for ( uint32_t j = 0; j < 8; ++j )
			{
				p[j] *= f.scale;
				p[j] = rotate( f.rotation, p[j] );
				p[j] += f.translation;
				box.grow( p[j] );
			}
		}

		embreePrims[i].primID  = i;
		embreePrims[i].lower_x = box.m_min.x;
		embreePrims[i].lower_y = box.m_min.y;
		embreePrims[i].lower_z = box.m_min.z;
		embreePrims[i].upper_x = box.m_max.x;
		embreePrims[i].upper_y = box.m_max.y;
		embreePrims[i].upper_z = box.m_max.z;
	}

	if ( embreePrims.size() > 1 )
		buildEmbreeBvh( embreeDevice, embreePrims, nullptr, buildInput.nodeList );
	else
		buildInput.nodeList.nodeCount = 1;
}

bool hiprtTest::readSourceCode(
	const std::filesystem::path& srcPath, std::string& sourceCode, std::vector<std::filesystem::path>* includes )
{
	std::fstream f( srcPath );
	if ( f.is_open() )
	{
		size_t sizeFile;
		f.seekg( 0, std::fstream::end );
		size_t size = sizeFile = static_cast<size_t>( f.tellg() );
		f.seekg( 0, std::fstream::beg );
		if ( includes != nullptr )
		{
			sourceCode.clear();
			std::string line;
			while ( std::getline( f, line ) )
			{
				if ( line.find( "#include" ) != std::string::npos )
				{
					size_t		pa	= line.find( "<" );
					size_t		pb	= line.find( ">" );
					std::string buf = line.substr( pa + 1, pb - pa - 1 );
					includes->push_back( buf );
				}
				sourceCode += line + '\n';
			}
		}
		else
		{
			sourceCode.resize( size, ' ' );
			f.read( &sourceCode[0], size );
		}
		f.close();
	}
	else
		return false;
	return true;
}

hiprtError hiprtTest::buildTraceKernels(
	hiprtContext								 ctxt,
	const std::filesystem::path&				 srcPath,
	std::vector<const char*>					 functionNames,
	std::vector<hiprtApiFunction>&				 functionsOut,
	std::optional<std::vector<const char*>>		 opts,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::vector<std::filesystem::path> includeNamesData;
	std::string						   sourceCode;
	readSourceCode( srcPath, sourceCode, &includeNamesData );

	std::vector<const char*> options;
	if ( opts ) options = *opts;

	options.push_back( "--use_fast_math" );

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<std::string> includeNameStrings( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	for ( size_t i = 0; i < includeNamesData.size(); i++ )
	{
		readSourceCode( getRootDir() / includeNamesData[i], headersData[i] );
		includeNameStrings[i] = includeNamesData[i].string();
		includeNames.push_back( includeNameStrings[i].c_str() );
		headers.push_back( headersData[i].c_str() );
	}

	functionsOut.resize( functionNames.size() );
	return hiprtBuildTraceKernels(
		ctxt,
		static_cast<uint32_t>( functionNames.size() ),
		functionNames.data(),
		sourceCode.c_str(),
		srcPath.string().c_str(),
		static_cast<uint32_t>( headers.size() ),
		headers.data(),
		includeNames.data(),
		static_cast<uint32_t>( options.size() ),
		options.data(),
		numGeomTypes,
		numRayTypes,
		funcNameSets ? funcNameSets.value().data() : nullptr,
		functionsOut.data(),
		nullptr,
		true );
}

hiprtError hiprtTest::buildTraceKernelsFromBitcode(
	hiprtContext								 ctxt,
	const std::filesystem::path&				 srcPath,
	std::vector<const char*>					 functionNames,
	std::vector<hiprtApiFunction>&				 functionsOut,
	std::optional<std::vector<const char*>>		 opts,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::vector<std::filesystem::path> includeNamesData;
	std::string						   sourceCode;
	readSourceCode( srcPath, sourceCode, &includeNamesData );

	std::vector<const char*> options;
	if ( opts ) options = *opts;

	std::string includePath = "-I" + getRootDir().string();
	options.push_back( includePath.c_str() );
	options.push_back( "--use_fast_math" );
	options.push_back( "--device-c" );
	options.push_back( "-std=c++17" );

	int major = 0;
	int minor = 0;
	checkOro( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, m_cudaDevice ) );
	checkOro( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, m_cudaDevice ) );
	std::string archOpt = "--gpu-architecture=compute_" + std::to_string( major ) + std::to_string( minor );
	options.push_back( archOpt.c_str() );

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<std::string> includeNameStrings( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	for ( size_t i = 0; i < includeNamesData.size(); i++ )
	{
		readSourceCode( getRootDir() / includeNamesData[i], headersData[i] );
		includeNameStrings[i] = includeNamesData[i].string();
		includeNames.push_back( includeNameStrings[i].c_str() );
		headers.push_back( headersData[i].c_str() );
	}

	nvrtcProgram prog = nullptr;
	checkOrortc( nvrtcCreateProgram(
		&prog,
		sourceCode.c_str(),
		srcPath.string().c_str(),
		static_cast<int>( headers.size() ),
		headers.data(),
		includeNames.data() ) );

	const nvrtcResult compileResult = nvrtcCompileProgram( prog, static_cast<int>( options.size() ), options.data() );
	if ( compileResult != NVRTC_SUCCESS )
	{
		size_t logSize = 0;
		checkOrortc( nvrtcGetProgramLogSize( prog, &logSize ) );
		if ( logSize != 0 )
		{
			std::string log( logSize, '\0' );
			checkOrortc( nvrtcGetProgramLog( prog, &log[0] ) );
			std::cerr << log << std::endl;
		}
		checkOrortc( nvrtcDestroyProgram( &prog ) );
		return hiprtErrorInternal;
	}

	std::string binary = getNvrtcCompiledBinaryForTest( prog );
	checkOrortc( nvrtcDestroyProgram( &prog ) );
	if ( binary.empty() )
		binary = compileSourceToCubinForTest( srcPath, options, getRootDir() );

	functionsOut.resize( functionNames.size() );
	return hiprtBuildTraceKernelsFromBitcode(
		ctxt,
		static_cast<uint32_t>( functionNames.size() ),
		functionNames.data(),
		srcPath.string().c_str(),
		binary.data(),
		binary.size(),
		numGeomTypes,
		numRayTypes,
		funcNameSets ? funcNameSets.value().data() : nullptr,
		functionsOut.data(),
		true );
}

hiprtError hiprtTest::buildTraceKernel(
	hiprtContext								 ctxt,
	const std::filesystem::path&				 srcPath,
	const std::string&							 functionName,
	cudaFunction_t&								 functionOut,
	std::optional<std::vector<const char*>>		 opts,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::vector<hiprtApiFunction> functions;
	hiprtError					  e =
		buildTraceKernels( ctxt, srcPath, { functionName.c_str() }, functions, opts, funcNameSets, numGeomTypes, numRayTypes );
	ASSERT( functions.size() == 1 );
	functionOut = *reinterpret_cast<cudaFunction_t*>( &functions.back() );
	return e;
}

hiprtError hiprtTest::buildTraceKernelFromBitcode(
	hiprtContext								 ctxt,
	const std::filesystem::path&				 srcPath,
	const std::string&							 functionName,
	cudaFunction_t&								 functionOut,
	std::optional<std::vector<const char*>>		 opts,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::vector<hiprtApiFunction> functions;
	hiprtError					  e = buildTraceKernelsFromBitcode(
		ctxt, srcPath, { functionName.c_str() }, functions, opts, funcNameSets, numGeomTypes, numRayTypes );
	ASSERT( functions.size() == 1 );
	functionOut = *reinterpret_cast<cudaFunction_t*>( &functions.back() );
	return e;
}

bool hiprtTest::loadBinaryFile( const std::filesystem::path& path, std::vector<uint8_t>& binary )
{
	std::ifstream file( path, std::ios::binary | std::ios::in | std::ios::ate );
	if ( !file.is_open() ) return false;

	const size_t size = static_cast<size_t>( file.tellg() );
	file.seekg( 0, std::ios::beg );
	binary.resize( size );
	file.read( reinterpret_cast<char*>( binary.data() ), static_cast<std::streamsize>( size ) );
	return true;
}

std::filesystem::path hiprtTest::findPrecompiledTraceKernelPath()
{
	const std::string filename = std::string( "hiprt" ) + HIPRT_VERSION_STR + "_nv_precompiled_bitcode.fatbin";
	const std::vector<std::filesystem::path> candidates = {
		getRootDir() / "dist/bin/Release" / filename,
		getRootDir() / "dist/bin/Debug" / filename,
		getRootDir() / "hiprt/bitcodes" / filename,
	};

	for ( const auto& candidate : candidates )
	{
		if ( std::filesystem::exists( candidate ) ) return candidate;
	}

	return {};
}

hiprtError hiprtTest::loadPrecompiledTraceKernel(
	const std::string& functionName, cudaFunction_t& functionOut, CUmodule* moduleOut )
{
	const std::filesystem::path path = findPrecompiledTraceKernelPath();
	if ( path.empty() ) return hiprtErrorInternal;

	std::vector<uint8_t> binary;
	if ( !loadBinaryFile( path, binary ) ) return hiprtErrorInternal;

	CUmodule module = nullptr;
	checkOro( cuModuleLoadData( &module, binary.data() ) );

	CUfunction function = nullptr;
	checkOro( cuModuleGetFunction( &function, module, functionName.c_str() ) );

	functionOut = function;
	if ( moduleOut ) *moduleOut = module;
	return hiprtSuccess;
}

void hiprtTest::createCornellTriangleMeshPrimitive(
	uint32_t triangleCount, hiprtTriangleMeshPrimitive& mesh, std::vector<void*>& garbageCollector )
{
	ASSERT( triangleCount > 0 && triangleCount <= CornellBoxTriangleCount );

	mesh = {};
	mesh.triangleCount	= triangleCount;
	mesh.triangleStride = sizeof( uint3 );
	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	garbageCollector.push_back( const_cast<void*>( mesh.triangleIndices ) );

	std::vector<uint32_t> indices( 3 * triangleCount );
	std::iota( indices.begin(), indices.end(), 0u );
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), reinterpret_cast<uint3*>( indices.data() ), mesh.triangleCount );

	mesh.vertexCount  = 3 * triangleCount;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	garbageCollector.push_back( const_cast<void*>( mesh.vertices ) );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), const_cast<float3*>( cornellBoxVertices.data() ), mesh.vertexCount );
}

void hiprtTest::createIndexedQuadStripTriangleMeshPrimitive(
	uint32_t quadCount, hiprtTriangleMeshPrimitive& mesh, std::vector<void*>& garbageCollector )
{
	ASSERT( quadCount > 0 );

	mesh = {};
	mesh.triangleCount	= 2 * quadCount;
	mesh.triangleStride = sizeof( uint3 );

	std::vector<uint3>  indices( mesh.triangleCount );
	std::vector<float3> vertices( 2 * ( quadCount + 1 ) );

	for ( uint32_t i = 0; i <= quadCount; ++i )
	{
		vertices[2 * i + 0] = { static_cast<float>( i ), 0.0f, 0.0f };
		vertices[2 * i + 1] = { static_cast<float>( i ), 1.0f, 0.0f };
	}

	for ( uint32_t i = 0; i < quadCount; ++i )
	{
		const uint32_t v0 = 2 * i + 0;
		const uint32_t v1 = 2 * i + 1;
		const uint32_t v2 = 2 * i + 2;
		const uint32_t v3 = 2 * i + 3;
		indices[2 * i + 0] = { v0, v1, v2 };
		indices[2 * i + 1] = { v2, v1, v3 };
	}

	malloc( reinterpret_cast<uint3*&>( mesh.triangleIndices ), mesh.triangleCount );
	garbageCollector.push_back( const_cast<void*>( mesh.triangleIndices ) );
	copyHtoD( reinterpret_cast<uint3*>( mesh.triangleIndices ), indices.data(), mesh.triangleCount );

	mesh.vertexCount  = static_cast<uint32_t>( vertices.size() );
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	garbageCollector.push_back( const_cast<void*>( mesh.vertices ) );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), vertices.data(), vertices.size() );
}

void hiprtTest::attachSequentialTrianglePairs(
	uint32_t pairCount, hiprtTriangleMeshPrimitive& mesh, std::vector<void*>& garbageCollector )
{
	ASSERT( pairCount > 0 );
	ASSERT( 2 * pairCount <= mesh.triangleCount );

	std::vector<uint2> pairIndices( pairCount );
	for ( uint32_t i = 0; i < pairCount; ++i )
		pairIndices[i] = { 2 * i + 0, 2 * i + 1 };

	mesh.trianglePairCount = pairCount;
	malloc( reinterpret_cast<uint2*&>( mesh.trianglePairIndices ), pairCount );
	garbageCollector.push_back( const_cast<void*>( mesh.trianglePairIndices ) );
	copyHtoD( reinterpret_cast<uint2*>( mesh.trianglePairIndices ), pairIndices.data(), pairIndices.size() );
}

void hiprtTest::destroyGarbage( std::vector<void*>& garbageCollector )
{
	for ( void* ptr : garbageCollector )
		free( ptr );
	garbageCollector.clear();
}

void hiprtTest::buildBatchGeometriesBuildOnly(
	const std::vector<hiprtGeometryBuildInput>& geomInputs, const hiprtBuildOptions& options )
{
	hiprtContext ctxt;
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );
	checkHiprt( hiprtSetLogLevel( ctxt, hiprtLogLevelError | hiprtLogLevelWarn ) );

	hiprtDevicePtr tempGeomBuffer = nullptr;
	size_t		   tempGeomSize   = 0;
	checkHiprt( hiprtGetGeometriesBuildTemporaryBufferSize(
		ctxt, static_cast<uint32_t>( geomInputs.size() ), geomInputs.data(), options, tempGeomSize ) );
	if ( tempGeomSize > 0 ) malloc( reinterpret_cast<uint8_t*&>( tempGeomBuffer ), tempGeomSize );

	std::vector<hiprtGeometry>	 geometries( geomInputs.size() );
	std::vector<hiprtGeometry*> geomAddrs( geomInputs.size() );
	for ( size_t i = 0; i < geometries.size(); ++i )
		geomAddrs[i] = &geometries[i];

	checkHiprt( hiprtCreateGeometries(
		ctxt, static_cast<uint32_t>( geomInputs.size() ), geomInputs.data(), options, geomAddrs.data() ) );
	checkHiprt( hiprtBuildGeometries(
		ctxt,
		hiprtBuildOperationBuild,
		static_cast<uint32_t>( geomInputs.size() ),
		geomInputs.data(),
		options,
		tempGeomBuffer,
		0,
		geometries.data() ) );

	if ( tempGeomBuffer != nullptr ) free( tempGeomBuffer );
	checkHiprt( hiprtDestroyGeometries( ctxt, static_cast<uint32_t>( geometries.size() ), geometries.data() ) );
	checkHiprt( hiprtDestroyContext( ctxt ) );
}

void hiprtTest::validateAndWriteImage(
	const std::filesystem::path& imgPath, uint8_t* data, std::optional<std::filesystem::path> refFilename )
{
	std::vector<uint8_t> image( g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	copyDtoH( image.data(), data, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	writeImage( imgPath, g_parsedArgs.m_ww, g_parsedArgs.m_wh, image.data() );

	if ( refFilename )
	{
		int refW;
		int refH;
		int refB;

		std::filesystem::path fullRefFilename = g_parsedArgs.m_referencePath / refFilename.value();
		uint8_t*			  ref			  = stbi_load( fullRefFilename.string().c_str(), &refW, &refH, &refB, 0 );
		if ( ref == 0 )
		{
			std::cerr << "Unable to open reference image '" << fullRefFilename << "'!" << std::endl;
			EXPECT_FALSE( 1 );
			return;
		}

		if ( static_cast<uint32_t>( refW ) != g_parsedArgs.m_ww || static_cast<uint32_t>( refH ) != g_parsedArgs.m_wh )
		{
			std::cerr << "Framebuffer resolution does not match!" << std::endl;
			EXPECT_FALSE( 1 );
			return;
		}

		uint32_t pixelThreshold = 10;
		uint32_t maxDiff		= 0;
		uint32_t nDiffPixels	= 0;

		for ( uint32_t i = 0; i < g_parsedArgs.m_ww * g_parsedArgs.m_wh; i++ )
		{
			uint32_t r = abs( image[i * 4 + 0] - ref[i * 4 + 0] );
			uint32_t g = abs( image[i * 4 + 1] - ref[i * 4 + 1] );
			uint32_t b = abs( image[i * 4 + 2] - ref[i * 4 + 2] );
			uint32_t a = abs( image[i * 4 + 3] - ref[i * 4 + 3] );

			if ( r > pixelThreshold || g > pixelThreshold || b > pixelThreshold || a > pixelThreshold )
			{
				maxDiff = std::max( maxDiff, std::max( r, std::max( g, b ) ) );
				nDiffPixels++;
			}
		}

		const float fail = 100.0f * nDiffPixels / ( static_cast<float>( g_parsedArgs.m_ww * g_parsedArgs.m_wh ) );
		if ( nDiffPixels != 0 )
			std::cerr << "Pixel difference: " << nDiffPixels << " (" << std::setprecision( 1 ) << fail
					  << "%)	(max diff: " << maxDiff << "/255)" << std::endl;

		if ( !( fail < 0.3f || maxDiff < 10 ) )
		{
			EXPECT_FALSE( 1 );
		}

		stbi_image_free( ref );
	}
}

void hiprtTest::writeImage( const std::filesystem::path& imgPath, uint32_t width, uint32_t height, uint8_t* data )
{
	stbi_write_png( imgPath.string().c_str(), width, height, 4, data, width * 4 );
}

void hiprtTest::launchKernel( cudaFunction_t func, uint32_t nx, uint32_t ny, void** args, uint32_t sharedMemoryBytes )
{
	constexpr uint32_t tx  = 16u;
	constexpr uint32_t ty  = 16u;
	uint32_t		   nbx = hiprt::DivideRoundUp( nx, tx );
	uint32_t		   nby = hiprt::DivideRoundUp( ny, ty );
	checkOro( cuLaunchKernel( func, nbx, nby, 1, tx, ty, 1, sharedMemoryBytes, 0, args, 0 ) ); // cudaLaunchKernel( func, dim3(nbx, nby, 1), dim3(tx, ty, 1), args, sharedMemoryBytes, 0 ) );
}

void hiprtTest::launchKernel(
	cudaFunction_t func, uint32_t nx, uint32_t ny, uint32_t tx, uint32_t ty, void** args, uint32_t sharedMemoryBytes )
{
	uint32_t nbx = hiprt::DivideRoundUp( nx, tx );
	uint32_t nby = hiprt::DivideRoundUp( ny, ty );
	checkOro( cuLaunchKernel( func, nbx, nby, 1, tx, ty, 1, sharedMemoryBytes, 0, args, 0 ) ); // cudaLaunchKernel( func, dim3(nbx, nby, 1), dim3(tx, ty, 1), args, sharedMemoryBytes, 0 ) );
}

void ObjTestCases::createScene(
	SceneData&					 scene,
	const std::filesystem::path& filename,
	bool						 enableRayMask,
	std::optional<hiprtFrameSRT> frame,
	hiprtBuildFlags				 bvhBuildFlag,
	bool						 time )
{
	checkHiprt( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, scene.m_ctx ) );
	checkHiprt( hiprtSetLogLevel( scene.m_ctx, hiprtLogLevelError | hiprtLogLevelWarn ) );

	tinyobj::attrib_t				 attrib;
	std::vector<tinyobj::shape_t>	 shapes;
	std::vector<tinyobj::material_t> materials;
	std::string						 err;
	std::string						 warning;

	bool ret = tinyobj::LoadObj(
		&attrib, &shapes, &materials, &warning, &err, filename.string().c_str(), filename.parent_path().string().c_str() );

	if ( !warning.empty() )
	{
		std::cerr << "OBJ Loader WARN : " << warning << std::endl;
	}

	if ( !err.empty() )
	{
		std::cerr << "OBJ Loader ERROR : " << err << std::endl;
		std::abort();
	}

	if ( !ret )
	{
		std::cerr << "Failed to load obj file" << std::endl;
		std::abort();
	}

	if ( shapes.empty() )
	{
		std::cerr << "No shapes in obj file (run 'git lfs fetch' and 'git lfs pull' in 'test/common/meshes/lfs')" << std::endl;
		std::abort();
	}

	std::vector<Material> shapeMaterials; // materials for all instances
	std::vector<Light>	  lights;
	std::vector<uint32_t> materialIndices; // material ids for all instances
	std::vector<uint32_t> instanceMask;
	std::vector<float3>	  allVertices;
	std::vector<float3>	  allNormals;
	std::vector<uint32_t> allIndices;
	std::vector<Aabb>	  geomBoxes;

	uint32_t numOfLights = 0;

	// Prefix sum to calculate the offsets in to global vert,index and material buffer
	uint32_t				 vertexPrefixSum = 0u;
	uint32_t				 normalPrefixSum = 0u;
	uint32_t				 indexPrefixSum	 = 0u;
	uint32_t				 matIdxPrefixSum = 0u;
	std::vector<uint32_t>	 indicesOffsets;
	std::vector<uint32_t>	 verticesOffsets;
	std::vector<uint32_t>	 normalsOffsets;
	std::vector<uint32_t>	 matIdxOffset;
	std::chrono::nanoseconds bvhBuildTime{};

	indicesOffsets.resize( shapes.size() );
	verticesOffsets.resize( shapes.size() );
	normalsOffsets.resize( shapes.size() );
	matIdxOffset.resize( shapes.size() );

	auto convert = []( const tinyobj::real_t c[3] ) -> float3 { return float3{ c[0], c[1], c[2] }; };

	for ( const auto& mat : materials )
	{
		Material m;
		m.m_diffuse	 = convert( mat.diffuse );
		m.m_emission = convert( mat.emission );
		shapeMaterials.push_back( m );
	}

	RTCDevice embreeDevice{};
	if ( ( bvhBuildFlag & 3 ) == hiprtBuildFlagBitCustomBvhImport )
	{
		embreeDevice = rtcNewDevice( "" );
		rtcSetDeviceErrorFunction(
			embreeDevice,
			[]( [[maybe_unused]] void* userPtr, [[maybe_unused]] enum RTCError code, const char* str ) {
				std::cerr << str << std::endl;
			},
			nullptr );
	}

	auto compare = []( const tinyobj::index_t& a, const tinyobj::index_t& b ) {
		if ( a.vertex_index < b.vertex_index ) return true;
		if ( a.vertex_index > b.vertex_index ) return false;

		if ( a.normal_index < b.normal_index ) return true;
		if ( a.normal_index > b.normal_index ) return false;

		if ( a.texcoord_index < b.texcoord_index ) return true;
		if ( a.texcoord_index > b.texcoord_index ) return false;

		return false;
	};

	for ( size_t i = 0; i < shapes.size(); ++i )
	{
		std::vector<float3>										  vertices;
		std::vector<float3>										  normals;
		std::vector<uint32_t>									  indices;
		float3*													  v = reinterpret_cast<float3*>( attrib.vertices.data() );
		std::map<tinyobj::index_t, uint32_t, decltype( compare )> knownIndex( compare );
		Aabb													  geomBox;

		for ( size_t face = 0; face < shapes[i].mesh.num_face_vertices.size(); face++ )
		{
			tinyobj::index_t idx0 = shapes[i].mesh.indices[3 * face + 0];
			tinyobj::index_t idx1 = shapes[i].mesh.indices[3 * face + 1];
			tinyobj::index_t idx2 = shapes[i].mesh.indices[3 * face + 2];

			if ( knownIndex.find( idx0 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx0] );
			}
			else
			{
				knownIndex[idx0] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx0] );
				vertices.push_back( v[idx0.vertex_index] );
				normals.push_back( v[idx0.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( knownIndex.find( idx1 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx1] );
			}
			else
			{
				knownIndex[idx1] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx1] );
				vertices.push_back( v[idx1.vertex_index] );
				normals.push_back( v[idx1.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( knownIndex.find( idx2 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx2] );
			}
			else
			{
				knownIndex[idx2] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx2] );
				vertices.push_back( v[idx2.vertex_index] );
				normals.push_back( v[idx2.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( !shapeMaterials.empty() && shapeMaterials[shapes[i].mesh.material_ids[face]].light() )
			{
				Light l;
				l.m_le = float3{
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.x + 40.f,
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.y + 40.f,
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.z + 40.f };

				size_t idx = indices.size() - 1;
				l.m_lv0	   = vertices[indices[idx - 2]];
				l.m_lv1	   = vertices[indices[idx - 1]];
				l.m_lv2	   = vertices[indices[idx - 0]];

				lights.push_back( l );
				numOfLights++;
			}

			materialIndices.push_back(
				shapes[i].mesh.material_ids[face] >= 0 ? shapes[i].mesh.material_ids[face] : hiprtInvalidValue );
		}

		verticesOffsets[i] = vertexPrefixSum;
		vertexPrefixSum += static_cast<uint32_t>( vertices.size() );
		indicesOffsets[i] = indexPrefixSum;
		indexPrefixSum += static_cast<uint32_t>( indices.size() );
		matIdxOffset[i] = matIdxPrefixSum;
		matIdxPrefixSum += static_cast<uint32_t>( shapes[i].mesh.material_ids.size() );
		normalsOffsets[i] = normalPrefixSum;
		normalPrefixSum += static_cast<uint32_t>( normals.size() );

		uint32_t mask = ~0u;
		if ( enableRayMask && ( i % 2 == 0 ) ) mask = 0u;

		instanceMask.push_back( mask );
		geomBoxes.push_back( geomBox );

		allVertices.insert( allVertices.end(), vertices.begin(), vertices.end() );
		allNormals.insert( allNormals.end(), normals.begin(), normals.end() );
		allIndices.insert( allIndices.end(), indices.begin(), indices.end() );
	}

	uint32_t threadCount = std::min( std::thread::hardware_concurrency(), 16u );
	if ( m_ctxtInput.deviceType != hiprtDeviceNVIDIA ) threadCount = 1;
	if ( ( bvhBuildFlag & 3 ) == hiprtBuildFlagBitCustomBvhImport ) threadCount = 1;
	std::vector<std::thread>			  threads( threadCount );
	std::vector<std::chrono::nanoseconds> bvhBuildTimes( threadCount );
	std::vector<cudaStream_t>				  streams( threadCount );
	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		checkOro( cudaStreamCreate( &streams[threadIndex] ) );
	}

	m_scene.m_geometries.resize( shapes.size() );
	m_scene.m_instances.resize( shapes.size() );
	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		threads[threadIndex] = std::thread(
			[&]( uint32_t threadIndex ) {

				std::vector<hiprtGeometry*>			 geomAddrs;
				std::vector<hiprtGeometryBuildInput> geomInputs;
				for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
				{
					hiprtTriangleMeshPrimitive mesh;

					uint32_t* indices	= &allIndices[indicesOffsets[i]];
					mesh.triangleCount	= static_cast<uint32_t>( shapes[i].mesh.num_face_vertices.size() );
					mesh.triangleStride = sizeof( uint32_t ) * 3;
					malloc( reinterpret_cast<uint8_t*&>( mesh.triangleIndices ), 3 * mesh.triangleCount * sizeof( uint32_t ) );
					copyHtoDAsync(
						reinterpret_cast<uint32_t*>( mesh.triangleIndices ),
						indices,
						3 * mesh.triangleCount,
						streams[threadIndex] );

					float3* vertices  = &allVertices[verticesOffsets[i]];
					mesh.vertexCount  = ( i + 1 == shapes.size() ) ? vertexPrefixSum - verticesOffsets[i]
																   : verticesOffsets[i + 1] - verticesOffsets[i];
					mesh.vertexStride = sizeof( float3 );
					malloc( reinterpret_cast<uint8_t*&>( mesh.vertices ), mesh.vertexCount * sizeof( float3 ) );
					copyHtoDAsync(
						reinterpret_cast<float3*>( mesh.vertices ), vertices, mesh.vertexCount, streams[threadIndex] );

					hiprtGeometryBuildInput geomInput;
					geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
					geomInput.primitive.triangleMesh = mesh;
					geomInput.geomType				 = 0;

					if ( ( bvhBuildFlag & 3 ) == hiprtBuildFlagBitCustomBvhImport )
						buildEmbreeGeometryBvh( embreeDevice, vertices, indices, geomInput );

					geomInputs.push_back( geomInput );
					geomAddrs.push_back( &m_scene.m_geometries[i] );
				}

				if ( !geomInputs.empty() )
				{
					hiprtBuildOptions options;
					options.buildFlags = bvhBuildFlag;

					size_t geomTempSize;
					checkHiprt( hiprtGetGeometriesBuildTemporaryBufferSize(
						scene.m_ctx, static_cast<uint32_t>( geomInputs.size() ), geomInputs.data(), options, geomTempSize ) );

					hiprtDevicePtr tempGeomBuffer = nullptr;
					if ( geomTempSize > 0 ) malloc( reinterpret_cast<uint8_t*&>( tempGeomBuffer ), geomTempSize );

					checkHiprt( hiprtCreateGeometries(
						scene.m_ctx,
						static_cast<uint32_t>( geomInputs.size() ),
						geomInputs.data(),
						options,
						geomAddrs.data() ) );

					std::vector<hiprtGeometry> geoms;
					for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
						geoms.push_back( m_scene.m_geometries[i] );

					std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
					checkHiprt( hiprtBuildGeometries(
						scene.m_ctx,
						hiprtBuildOperationBuild,
						static_cast<uint32_t>( geomInputs.size() ),
						geomInputs.data(),
						options,
						tempGeomBuffer,
						streams[threadIndex],
						geoms.data() ) );
					std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
					bvhBuildTimes[threadIndex] += end - begin;

					size_t j = 0;
					for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
					{
						m_scene.m_geometries[i]			= geoms[j++];
						m_scene.m_instances[i].type		= hiprtInstanceTypeGeometry;
						m_scene.m_instances[i].geometry = m_scene.m_geometries[i];
					}

					for ( auto& geomInput : geomInputs )
					{
						free( geomInput.primitive.triangleMesh.triangleIndices );
						free( geomInput.primitive.triangleMesh.vertices );
						if ( ( bvhBuildFlag & 3 ) == hiprtBuildFlagBitCustomBvhImport )
						{
							free( geomInput.nodeList.leafNodes );
							free( geomInput.nodeList.internalNodes );
							free( geomInput.primitive.triangleMesh.trianglePairIndices );
						}
					}

					if ( geomTempSize > 0 ) free( tempGeomBuffer );

					waitForCompletion( streams[threadIndex] );
				}
			},
			threadIndex );
	}

	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		threads[threadIndex].join();
		checkOro( cudaStreamDestroy( streams[threadIndex] ) );
		bvhBuildTime = std::max( bvhBuildTime, bvhBuildTimes[threadIndex] );
	}

	// copy vertex offset
	malloc( scene.m_vertexOffsets, verticesOffsets.size() );
	copyHtoD( scene.m_vertexOffsets, verticesOffsets.data(), verticesOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_vertexOffsets );

	// copy normals
	malloc( scene.m_normals, allNormals.size() );
	copyHtoD( scene.m_normals, allNormals.data(), allNormals.size() );
	scene.m_garbageCollector.push_back( scene.m_normals );

	// copy normal offsets
	malloc( scene.m_normalOffsets, normalsOffsets.size() );
	copyHtoD( scene.m_normalOffsets, normalsOffsets.data(), normalsOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_normalOffsets );

	// copy indices
	malloc( scene.m_indices, allIndices.size() );
	copyHtoD( scene.m_indices, allIndices.data(), allIndices.size() );
	scene.m_garbageCollector.push_back( scene.m_indices );

	// copy index offsets
	malloc( scene.m_indexOffsets, indicesOffsets.size() );
	copyHtoD( scene.m_indexOffsets, indicesOffsets.data(), indicesOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_indexOffsets );

	// copy material indices
	malloc( scene.m_bufMaterialIndices, materialIndices.size() );
	copyHtoD( scene.m_bufMaterialIndices, materialIndices.data(), materialIndices.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMaterialIndices );

	// copy material offset
	malloc( scene.m_bufMatIdsPerInstance, matIdxOffset.size() );
	copyHtoD( scene.m_bufMatIdsPerInstance, matIdxOffset.data(), matIdxOffset.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMatIdsPerInstance );

	// copy materials
	if ( shapeMaterials.empty() )
	{ // default material to prevent crash
		Material mat;
		mat.m_diffuse  = hiprt::make_float3( 1.0f );
		mat.m_emission = hiprt::make_float3( 0.0f );
		shapeMaterials.push_back( mat );
	}
	malloc( scene.m_bufMaterials, shapeMaterials.size() );
	copyHtoD( scene.m_bufMaterials, shapeMaterials.data(), shapeMaterials.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMaterials );

	// copy light
	if ( !lights.empty() )
	{
		malloc( scene.m_lights, lights.size() );
		copyHtoD( scene.m_lights, lights.data(), lights.size() );
		scene.m_garbageCollector.push_back( scene.m_lights );
	}

	// copy light num
	malloc( scene.m_numOfLights, 1 );
	copyHtoD( scene.m_numOfLights, &numOfLights, 1 );
	scene.m_garbageCollector.push_back( scene.m_numOfLights );

	// prepare scene
	hiprtScene			 sceneLocal;
	hiprtDevicePtr		 sceneTemp = nullptr;
	hiprtSceneBuildInput sceneInput{};
	{
		sceneInput.instanceCount = static_cast<uint32_t>( shapes.size() );
		malloc( reinterpret_cast<uint32_t*&>( sceneInput.instanceMasks ), sceneInput.instanceCount );
		copyHtoD( reinterpret_cast<uint32_t*>( sceneInput.instanceMasks ), instanceMask.data(), sceneInput.instanceCount );
		scene.m_garbageCollector.push_back( sceneInput.instanceMasks );

		malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
		copyHtoD(
			reinterpret_cast<hiprtInstance*>( sceneInput.instances ), m_scene.m_instances.data(), sceneInput.instanceCount );
		scene.m_garbageCollector.push_back( sceneInput.instances );

		std::vector<hiprtFrameSRT> frames;
		if ( frame )
		{
			sceneInput.frameCount				= sceneInput.instanceCount;
			sceneInput.instanceTransformHeaders = nullptr;

			for ( uint32_t i = 0; i < sceneInput.instanceCount; i++ )
				frames.push_back( frame.value() );

			malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), frames.size() );
			copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), frames.data(), frames.size() );
			scene.m_garbageCollector.push_back( sceneInput.instanceFrames );
		}

		if ( ( bvhBuildFlag & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			buildEmbreeSceneBvh( embreeDevice, geomBoxes, frames, sceneInput );
			scene.m_garbageCollector.push_back( sceneInput.nodeList.leafNodes );
			scene.m_garbageCollector.push_back( sceneInput.nodeList.internalNodes );
		}

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = bvhBuildFlag;
		checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( scene.m_ctx, sceneInput, options, sceneTempSize ) );
		if ( sceneTempSize > 0 )
		{
			malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );
			scene.m_garbageCollector.push_back( sceneTemp );
		}

		checkHiprt( hiprtCreateScene( scene.m_ctx, sceneInput, options, sceneLocal ) );

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		checkHiprt( hiprtBuildScene( scene.m_ctx, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, sceneLocal ) );
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		bvhBuildTime += ( end - begin );

		if ( time )
			std::cout << "Bvh build time : " << std::chrono::duration_cast<std::chrono::milliseconds>( bvhBuildTime ).count()
					  << " ms" << std::endl;
		scene.m_scene = sceneLocal;
	}

	if ( ( bvhBuildFlag & 3 ) == hiprtBuildFlagBitCustomBvhImport ) rtcReleaseDevice( embreeDevice );
}

void ObjTestCases::setupScene(
	Camera&						 camera,
	const std::filesystem::path& filename,
	bool						 enableRayMask,
	std::optional<hiprtFrameSRT> frame,
	hiprtBuildFlags				 bvhBuildFlag,
	bool						 time )
{
	m_camera = camera;
	createScene( m_scene, filename, enableRayMask, frame, bvhBuildFlag, time );
}

void ObjTestCases::deleteScene( SceneData& scene )
{
	checkHiprt( hiprtDestroyScene( scene.m_ctx, scene.m_scene ) );
	checkHiprt(
		hiprtDestroyGeometries( scene.m_ctx, static_cast<uint32_t>( scene.m_geometries.size() ), scene.m_geometries.data() ) );

	for ( void* ptr : scene.m_garbageCollector )
		free( ptr );

	checkHiprt( hiprtDestroyContext( scene.m_ctx ) );

	scene.m_bufMaterialIndices	 = nullptr;
	scene.m_bufMatIdsPerInstance = nullptr;
	scene.m_bufMaterials		 = nullptr;
	scene.m_vertices			 = nullptr;
	scene.m_vertexOffsets		 = nullptr;
	scene.m_normals				 = nullptr;
	scene.m_normalOffsets		 = nullptr;
	scene.m_indices				 = nullptr;
	scene.m_indexOffsets		 = nullptr;
	scene.m_lights				 = nullptr;
	scene.m_numOfLights			 = nullptr;
	scene.m_scene				 = nullptr;
	scene.m_ctx					 = nullptr;
	scene.m_geometries.clear();
	scene.m_instances.clear();
	scene.m_garbageCollector.clear();
}

void ObjTestCases::render(
	std::optional<std::filesystem::path> imgPath,
	const std::filesystem::path&		 kernelPath,
	const std::string&					 funcName,
	std::optional<std::filesystem::path> refFilename,
	bool								 time,
	float								 aoRadius )
{
	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	m_scene.m_garbageCollector.push_back( dst );

	uint32_t	   stackSize		  = 64u;
	const uint32_t sharedStackSize	  = 16u;
	const uint32_t blockWidth		  = 8u;
	const uint32_t blockHeight		  = 8u;
	const uint32_t blockSize		  = blockWidth * blockHeight;
	std::string	   blockSizeDef		  = "-DBLOCK_SIZE=" + std::to_string( blockSize );
	std::string	   sharedStackSizeDef = "-DSHARED_STACK_SIZE=" + std::to_string( sharedStackSize );

	std::vector<const char*> opts;
	opts.push_back( blockSizeDef.c_str() );
	opts.push_back( sharedStackSizeDef.c_str() );
	// opts.push_back( "-G" );

	hiprtGlobalStackBufferInput stackBufferInput{
		hiprtStackTypeGlobal,
		hiprtStackEntryTypeInteger,
		stackSize,
		static_cast<uint32_t>( g_parsedArgs.m_ww * g_parsedArgs.m_wh ) };
	if constexpr ( UseDynamicStack ) stackBufferInput.type = hiprtStackTypeDynamic;
	hiprtGlobalStackBuffer stackBuffer;
	checkHiprt( hiprtCreateGlobalStackBuffer( m_scene.m_ctx, stackBufferInput, stackBuffer ) );

	cudaFunction_t	   func;
	hiprtFuncTable funcTable = nullptr;

	if constexpr ( UseFilter )
	{
		hiprtFuncNameSet funcNameSet;
		funcNameSet.filterFuncName				   = "filter";
		std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

		hiprtFuncDataSet funcDataSet;
		checkHiprt( hiprtCreateFuncTable( m_scene.m_ctx, 1, 1, funcTable ) );
		checkHiprt( hiprtSetFuncTable( m_scene.m_ctx, funcTable, 0, 0, funcDataSet ) );
		buildTraceKernel( m_scene.m_ctx, kernelPath, funcName, func, opts, funcNameSets, 1, 1 );
	}
	else
	{
		buildTraceKernel( m_scene.m_ctx, kernelPath, funcName, func, opts );
	}

	uint2 res	 = { g_parsedArgs.m_ww, g_parsedArgs.m_wh };
	void* args[] = {
		&m_scene.m_scene,
		&dst,
		&res,
		&stackBuffer,
		&m_camera,
		&m_scene.m_bufMaterialIndices,
		&m_scene.m_bufMaterials,
		&m_scene.m_bufMatIdsPerInstance,
		&m_scene.m_indices,
		&m_scene.m_indexOffsets,
		&m_scene.m_normals,
		&m_scene.m_normalOffsets,
		&m_scene.m_numOfLights,
		&m_scene.m_lights,
		&aoRadius,
		&funcTable };

	// cudaFuncAttributes attr;
	// checkOro( cudaFuncGetAttributes( &attr, (const void*)func ) );
	// int numRegs{};
	// numRegs = attr.numRegs;
	// int numSmem{};
	// numSmem = attr.sharedSizeBytes;

	int numRegs{};
	checkOro( cuFuncGetAttribute( &numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, func ) );

	int numSmem{};
	checkOro( cuFuncGetAttribute( &numSmem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func ) );

	std::cout << "Trace kernel: registers " << numRegs << ", shared memory " << numSmem << std::endl;
	waitForCompletion();
	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, blockWidth, blockHeight, args );

	waitForCompletion();
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	checkHiprt( hiprtDestroyGlobalStackBuffer( m_scene.m_ctx, stackBuffer ) );
	if constexpr ( UseFilter ) checkHiprt( hiprtDestroyFuncTable( m_scene.m_ctx, funcTable ) );

	if ( time )
		std::cout << "Ray cast time: " << std::chrono::duration_cast<std::chrono::milliseconds>( end - begin ).count() << " ms"
				  << std::endl;

	if ( imgPath ) validateAndWriteImage( imgPath.value(), dst, refFilename );
}
