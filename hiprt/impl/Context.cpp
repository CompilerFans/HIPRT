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

#include <hiprt/impl/BvhCommon.h>
#include <hiprt/impl/BvhImporter.h>
#include <hiprt/impl/BatchBuilder.h>
#include <hiprt/impl/Context.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/LbvhBuilder.h>
#include <hiprt/impl/PlocBuilder.h>
#include <hiprt/impl/SbvhBuilder.h>
#include <hiprt/impl/Transform.h>

namespace hiprt
{
namespace
{
hiprtBuildFlags getEffectiveBuildFlags( const Context& context, hiprtBuildFlags buildFlags )
{
	const hiprtBuildFlags buildMode = static_cast<hiprtBuildFlags>( buildFlags & 3 );
	if ( buildMode == hiprtBuildFlagBitPreferHighQualityBuild && context.getDeviceName().find( "NVIDIA" ) == std::string::npos )
	{
		const hiprtBuildFlags preservedFlags = static_cast<hiprtBuildFlags>( buildFlags & ~3 );
		return static_cast<hiprtBuildFlags>( preservedFlags | hiprtBuildFlagBitPreferFastBuild );
	}

	return buildFlags;
}

void patchSceneInstanceNodes(
	const Context& context, const hiprtSceneBuildInput& buildInput, hiprtScene scene, cudaStream_t stream )
{
	if ( context.getDeviceName().find( "NVIDIA" ) != std::string::npos ) return;
	if ( buildInput.instanceCount == 0 || buildInput.frameCount == 0 ) return;

	SceneHeader header{};
	checkOro( cuMemcpyDtoH( &header, reinterpret_cast<size_t>( scene ), sizeof( SceneHeader ) ) );
	if ( header.m_primCount == 0 || header.m_primNodes == nullptr ) return;

	std::vector<hiprtTransformHeader> transformHeaders( buildInput.instanceCount );
	if ( buildInput.instanceTransformHeaders != nullptr )
	{
		checkOro( cuMemcpyDtoH(
			transformHeaders.data(),
			reinterpret_cast<size_t>( buildInput.instanceTransformHeaders ),
			sizeof( hiprtTransformHeader ) * buildInput.instanceCount ) );
	}
	else
	{
		for ( uint32_t i = 0; i < buildInput.instanceCount; ++i )
			transformHeaders[i] = hiprtTransformHeader{ i, 1u };
	}

	if ( context.getRtip() >= 31 )
	{
		std::vector<HwInstanceNode> nodes( header.m_primCount );
		checkOro( cuMemcpyDtoH(
			nodes.data(), reinterpret_cast<size_t>( header.m_primNodes ), sizeof( HwInstanceNode ) * header.m_primCount ) );
		for ( uint32_t i = 0; i < std::min<uint32_t>( header.m_primCount, buildInput.instanceCount ); ++i )
		{
			nodes[i].m_static   = 0;
			nodes[i].m_identity = 0;
			nodes[i].m_transform = transformHeaders[i];
		}
		checkOro( cuMemcpyHtoD(
			reinterpret_cast<size_t>( header.m_primNodes ), nodes.data(), sizeof( HwInstanceNode ) * header.m_primCount ) );
	}
	else
	{
		std::vector<UserInstanceNode> nodes( header.m_primCount );
		checkOro( cuMemcpyDtoH(
			nodes.data(), reinterpret_cast<size_t>( header.m_primNodes ), sizeof( UserInstanceNode ) * header.m_primCount ) );
		for ( uint32_t i = 0; i < std::min<uint32_t>( header.m_primCount, buildInput.instanceCount ); ++i )
		{
			nodes[i].m_static   = 0;
			nodes[i].m_identity = 0;
			nodes[i].m_transform = transformHeaders[i];
		}
		checkOro( cuMemcpyHtoD(
			reinterpret_cast<size_t>( header.m_primNodes ), nodes.data(), sizeof( UserInstanceNode ) * header.m_primCount ) );
	}

	if ( stream != 0 ) checkOro( cudaStreamSynchronize( stream ) );
}
} // namespace

Context::Context( const hiprtContextCreationInput& input )
{
	m_device = input.device;
	if ( m_device < 0 ) checkOro( cudaGetDevice( &m_device ) );

	checkOro( cudaSetDevice( m_device ) );

	CUdevice cuDevice = 0;
	checkOro( cuDeviceGet( &cuDevice, m_device ) );
	checkOro( cuDevicePrimaryCtxRetain( &m_ctxt, cuDevice ) );
	checkOro( cuCtxSetCurrent( m_ctxt ) );
}

Context::~Context()
{
	if ( m_ctxt == nullptr ) return;

	cuCtxSetCurrent( m_ctxt );
	m_compiler.clear();
	cuCtxSetCurrent( nullptr );

	CUdevice cuDevice = 0;
	if ( cuDeviceGet( &cuDevice, m_device ) == CUDA_SUCCESS )
	{
		cuDevicePrimaryCtxRelease( cuDevice );
	}
}

std::vector<hiprtGeometry>
Context::createGeometries( const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	const hiprtBuildFlags effectiveBuildFlags = getEffectiveBuildFlags( *this, buildOptions.buildFlags );
	const bool			 useBatchPath		= buildInputs.size() > 1;

	size_t				size = 0;
	std::vector<size_t> sizes( buildInputs.size() );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( useBatchPath && batchBuild( buildInputs[i], buildOptions ) )
		{
			logInfo( "BatchBuild::createGeometry\n" );
			sizes[i] = BatchBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			logInfo( "CustomBvhImport::createGeometry\n" );
			sizes[i] = BvhImporter::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
		{
			logInfo( "FastBuild::createGeometry\n" );
			sizes[i] = LbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
		{
			logInfo( "HighQualityBuild::createGeometry\n" );
			sizes[i] = SbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
		{
			logInfo( "BalancedBuild::createGeometry\n" );
			sizes[i] = PlocBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else
		{
			logWarn( "Unknow build option => FastBuild::createGeometry used instead\n" );
			sizes[i] = LbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
	}

	cudaDeviceptr buffer;
	checkOro( cudaMalloc( reinterpret_cast<void **>( reinterpret_cast<uintptr_t>(&buffer)), size ) );

	std::vector<hiprtGeometry> geometries( buildInputs.size() );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		geometries[i] = reinterpret_cast<hiprtGeometry>( buffer );
		buffer		  = reinterpret_cast<uint8_t*>( buffer ) + sizes[i];
	}

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	m_poolHeads[{ reinterpret_cast<cudaDeviceptr>( geometries.front() ), size }] = static_cast<uint32_t>( geometries.size() );

	return geometries;
}

void Context::destroyGeometries( const std::vector<hiprtGeometry> geometries )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	for ( hiprtGeometry geometry : geometries )
	{
		auto head = std::find_if(
			m_poolHeads.begin(), m_poolHeads.end(), [&]( const std::pair<std::pair<cudaDeviceptr, size_t>, uint32_t>& h ) {
				return reinterpret_cast<hiprtGeometry>( h.first.first ) <= geometry &&
					   reinterpret_cast<uint8_t*>( geometry ) < reinterpret_cast<uint8_t*>( h.first.first ) + h.first.second;
			} );

		if ( head != m_poolHeads.end() )
		{
			if ( --head->second == 0 )
			{
				checkOro( cudaFree( reinterpret_cast<void*>(head->first.first) ) );
				logInfo( "Geometry pool deallocated\n" );
				m_poolHeads.erase( head );
			}
		}
		else
		{
			logWarn( "Trying to destroy a geometry not allocated in this context!\n" );
		}
	}
}

void Context::buildGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	cudaStream_t									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	const hiprtBuildFlags effectiveBuildFlags = getEffectiveBuildFlags( *this, buildOptions.buildFlags );
	const bool			 useBatchPath		= buildInputs.size() > 1;

	std::vector<hiprtGeometryBuildInput> batchInputs;
	std::vector<hiprtDevicePtr>			 batchBuffers;
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( useBatchPath && batchBuild( buildInputs[i], buildOptions ) )
		{
			batchInputs.push_back( buildInputs[i] );
			batchBuffers.push_back( buffers[i] );
		}
	}

	if ( !batchInputs.empty() )
	{
		logInfo( "BatchBuild::buildGeometry\n" );
		BatchBuilder::build( *this, batchInputs, buildOptions, temporaryBuffer, stream, batchBuffers );
	}

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( !( useBatchPath && batchBuild( buildInputs[i], buildOptions ) ) )
		{
			if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
			{
				logInfo( "CustomBvhImport::buildGeometry\n" );
				BvhImporter::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
			{
				logInfo( "FastBuild::buildGeometry\n" );
				LbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
			{
				logInfo( "HighQualityBuild::buildGeometry\n" );
				SbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
			{
				logInfo( "BalancedBuild::buildGeometry\n" );
				PlocBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else
			{
				logWarn( "Unknow build option => FastBuild::buildGeometry used instead\n" );
				LbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
		}
	}
}

void Context::updateGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	cudaStream_t									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	const hiprtBuildFlags effectiveBuildFlags = getEffectiveBuildFlags( *this, buildOptions.buildFlags );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			logInfo( "CustomBvhImport::updateGeometry\n" );
			BvhImporter::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
		{
			logInfo( "FastBuild::updateGeometry\n" );
			LbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
		{
			logInfo( "HighQualityBuild::updateGeometry\n" );
			SbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
		{
			logInfo( "BalancedBuild::updateGeometry\n" );
			PlocBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else
		{
			logWarn( "Unknow build option => FastBuild::updateGeometry used instead\n" );
			LbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
	}
}

size_t Context::getGeometriesBuildTempBufferSize(
	const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	const hiprtBuildFlags effectiveBuildFlags = getEffectiveBuildFlags( *this, buildOptions.buildFlags );
	const bool			 useBatchPath		= buildInputs.size() > 1;
	std::vector<hiprtGeometryBuildInput> batchInputs;
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( useBatchPath && batchBuild( buildInputs[i], buildOptions ) ) batchInputs.push_back( buildInputs[i] );
	}

	size_t size = 0;
	if ( !batchInputs.empty() )
	{
		logInfo( "BatchBuild::getGeometryBuildTempBufferSize\n" );
		size = BatchBuilder::getTemporaryBufferSize( *this, batchInputs, buildOptions );
	}

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( !( useBatchPath && batchBuild( buildInputs[i], buildOptions ) ) )
		{
			if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
			{
				logInfo( "CustomBvhImport::getGeometryBuildTempBufferSize\n" );
				size = std::max( size, BvhImporter::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
			{
				logInfo( "FastBuild::getGeometryBuildTempBufferSize\n" );
				size = std::max( size, LbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
			{
				logInfo( "HighQualityBuild::getGeometryBuildTempBufferSize\n" );
				size = std::max( size, SbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
			{
				logInfo( "BalancedBuild::getGeometryBuildTempBufferSize\n" );
				size = std::max( size, PlocBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
			else
			{
				logWarn( "Unknow build option => FastBuild::getGeometryBuildTempBufferSize used instead\n" );
				size = std::max( size, LbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
		}
	}

	return size;
}

std::vector<hiprtGeometry> Context::compactGeometries( const std::vector<hiprtGeometry>& geometriesIn, cudaStream_t stream )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );

	size_t				size = 0;
	std::vector<size_t> sizes( geometriesIn.size() );
	for ( size_t i = 0; i < geometriesIn.size(); ++i )
	{
		GeomHeader header;
		checkOro( cuMemcpyDtoH( &header, reinterpret_cast<size_t>( geometriesIn[i] ), sizeof( GeomHeader ) ) );
		const size_t primNodeSize = header.m_geomType & 1 ? getTriangleNodeSize() : sizeof( CustomNode );
		sizes[i] =
			getGeometryStorageBufferSize( header.m_primNodeCount, header.m_boxNodeCount, primNodeSize, getBoxNodeSize() );
		size += sizes[i];
	}

	cudaDeviceptr buffer;
	checkOro( cudaMalloc( &buffer, size ) );

	std::vector<hiprtGeometry> geometriesOut( geometriesIn.size() );
	for ( size_t i = 0; i < geometriesIn.size(); ++i )
	{
		GeomHeader header;
		checkOro( cuMemcpyDtoH( &header, reinterpret_cast<size_t>( geometriesIn[i] ), sizeof( GeomHeader ) ) );
		const size_t primNodeSize = header.m_geomType & 1 ? getTriangleNodeSize() : sizeof( CustomNode );
		const size_t boxNodeSize  = getBoxNodeSize();

		geometriesOut[i] = reinterpret_cast<hiprtGeometry>( buffer );
		MemoryArena storageMemoryArena( geometriesOut[i], sizes[i], DefaultAlignment );
		storageMemoryArena.allocate<GeomHeader>();
		void* boxNodes	= storageMemoryArena.allocate<uint8_t>( boxNodeSize * header.m_boxNodeCount );
		void* primNodes = storageMemoryArena.allocate<uint8_t>( primNodeSize * header.m_primNodeCount );

		checkOro( cuMemcpyDtoDAsync(
			reinterpret_cast<size_t>( boxNodes ),
			reinterpret_cast<size_t>( header.m_boxNodes ),
			boxNodeSize * header.m_boxNodeCount,
			stream ) );

		checkOro( cuMemcpyDtoDAsync(
			reinterpret_cast<size_t>( primNodes ),
			reinterpret_cast<size_t>( header.m_primNodes ),
			primNodeSize * header.m_primNodeCount,
			stream ) );

		header.m_boxNodes  = boxNodes;
		header.m_primNodes = primNodes;
		checkOro(
			cuMemcpyHtoDAsync( reinterpret_cast<size_t>( geometriesOut[i] ), &header, sizeof( GeomHeader ), stream ) );

		buffer = reinterpret_cast<uint8_t*>( buffer ) + sizes[i];
	}

	{
		std::lock_guard<std::mutex> lockMutex( m_poolMutex );
		m_poolHeads[{ reinterpret_cast<void *>( geometriesOut.front() ), size }] =
			static_cast<uint32_t>( geometriesOut.size() );
	}

	checkOro( cudaStreamSynchronize( stream ) );
	destroyGeometries( geometriesIn );

	return geometriesOut;
}

std::vector<hiprtScene>
Context::createScenes( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	const hiprtBuildFlags effectiveBuildFlags = getEffectiveBuildFlags( *this, buildOptions.buildFlags );
	const bool			 useBatchPath		= buildInputs.size() > 1;

	size_t				size = 0;
	std::vector<size_t> sizes( buildInputs.size() );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( InstanceIDBits < 32 && buildInputs[i].instanceCount >= ( 1u << InstanceIDBits ) )
		{
			std::string msg = Utility::format(
				"The number of instances is %u is greater than the maximum supported number of instances (%u)",
				buildInputs[i].instanceCount,
				1u << InstanceIDBits );
			throw std::runtime_error( msg );
		}

		if ( useBatchPath && batchBuild( buildInputs[i], buildOptions ) )
		{
			logInfo( "BatchBuild::createScene\n" );
			sizes[i] = BatchBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			logInfo( "CustomBvhImport::createScene\n" );
			sizes[i] = BvhImporter::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
		{
			logInfo( "FastBuild::createScene\n" );
			sizes[i] = LbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
		{
			logInfo( "HighQualityBuild::createScene\n" );
			sizes[i] = SbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
		{
			logInfo( "BalancedBuild::createScene\n" );
			sizes[i] = PlocBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else
		{
			logWarn( "Unknow build option => FastBuild::createScene used instead\n" );
			sizes[i] = LbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
	}

	cudaDeviceptr buffer;
	checkOro( cudaMalloc( &buffer, size ) );

	std::vector<hiprtScene> scenes( buildInputs.size() );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		scenes[i] = reinterpret_cast<hiprtScene>( buffer );
		buffer	  = static_cast<uint8_t*>( buffer ) + sizes[i];
	}

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	m_poolHeads[{ reinterpret_cast<cudaDeviceptr>( scenes.front() ), size }] = static_cast<uint32_t>( scenes.size() );

	return scenes;
}

void Context::destroyScenes( const std::vector<hiprtScene> scenes )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	for ( hiprtScene scene : scenes )
	{
		auto head = std::find_if(
			m_poolHeads.begin(), m_poolHeads.end(), [&]( const std::pair<std::pair<cudaDeviceptr, size_t>, uint32_t>& h ) {
				return reinterpret_cast<hiprtScene>( h.first.first ) <= scene &&
					   reinterpret_cast<uint8_t*>( scene ) < reinterpret_cast<uint8_t*>( h.first.first ) + h.first.second;
			} );

		if ( head != m_poolHeads.end() )
		{
			if ( --head->second == 0 )
			{
				checkOro( cudaFree( reinterpret_cast<void*>(head->first.first) ) );
				logInfo( "Scene pool deallocated\n" );
				m_poolHeads.erase( head );
			}
		}
		else
		{
			logWarn( "Trying to destroy a scene not allocated in this context!\n" );
		}
	}
}

void Context::buildScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const hiprtBuildOptions					 buildOptions,
	hiprtDevicePtr							 temporaryBuffer,
	cudaStream_t								 stream,
	std::vector<hiprtDevicePtr>&			 buffers )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	const hiprtBuildFlags effectiveBuildFlags = getEffectiveBuildFlags( *this, buildOptions.buildFlags );
	const bool			 useBatchPath		= buildInputs.size() > 1;

	std::vector<hiprtSceneBuildInput> batchInputs;
	std::vector<hiprtDevicePtr>		  batchBuffers;
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( InstanceIDBits < 32 && buildInputs[i].instanceCount >= ( 1u << InstanceIDBits ) )
		{
			std::string msg = Utility::format(
				"The number of instances is %u is greater than the maximum supported number of instances (%u)",
				buildInputs[i].instanceCount,
				1u << InstanceIDBits );
			throw std::runtime_error( msg );
		}

		if ( useBatchPath && batchBuild( buildInputs[i], buildOptions ) )
		{
			batchInputs.push_back( buildInputs[i] );
			batchBuffers.push_back( buffers[i] );
		}
	}

	if ( !batchInputs.empty() )
	{
		logInfo( "BatchBuild::buildScene\n" );
		BatchBuilder::build( *this, batchInputs, buildOptions, temporaryBuffer, stream, batchBuffers );
	}

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( !( useBatchPath && batchBuild( buildInputs[i], buildOptions ) ) )
		{
			if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
			{
				logInfo( "CustomBvhImport::buildScene\n" );
				BvhImporter::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
				patchSceneInstanceNodes( *this, buildInputs[i], reinterpret_cast<hiprtScene>( buffers[i] ), stream );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
			{
				logInfo( "FastBuild::buildScene\n" );
				LbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
				patchSceneInstanceNodes( *this, buildInputs[i], reinterpret_cast<hiprtScene>( buffers[i] ), stream );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
			{
				logInfo( "HighQualityBuild::buildScene\n" );
				SbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
				patchSceneInstanceNodes( *this, buildInputs[i], reinterpret_cast<hiprtScene>( buffers[i] ), stream );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
			{
				logInfo( "BalancedBuild::buildScene\n" );
				PlocBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
				patchSceneInstanceNodes( *this, buildInputs[i], reinterpret_cast<hiprtScene>( buffers[i] ), stream );
			}
			else
			{
				logWarn( "Unknow build option => FastBuild::buildScene used instead\n" );
				LbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
				patchSceneInstanceNodes( *this, buildInputs[i], reinterpret_cast<hiprtScene>( buffers[i] ), stream );
			}
		}
	}
}

void Context::updateScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const hiprtBuildOptions					 buildOptions,
	hiprtDevicePtr							 temporaryBuffer,
	cudaStream_t								 stream,
	std::vector<hiprtDevicePtr>&			 buffers )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	const hiprtBuildFlags effectiveBuildFlags = getEffectiveBuildFlags( *this, buildOptions.buildFlags );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( InstanceIDBits < 32 && buildInputs[i].instanceCount >= ( 1u << InstanceIDBits ) )
		{
			std::string msg = Utility::format(
				"The number of instances is %u is greater than the maximum supported number of instances (%u)",
				buildInputs[i].instanceCount,
				1u << InstanceIDBits );
			throw std::runtime_error( msg );
		}

		if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			logInfo( "CustomBvhImport::updateScene\n" );
			BvhImporter::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
		{
			logInfo( "FastBuild::updateScene\n" );
			LbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
		{
			logInfo( "HighQualityBuild::updateScene\n" );
			SbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
		{
			logInfo( "BalancedBuild::updateScene\n" );
			PlocBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else
		{
			logWarn( "Unknow build option => FastBuild::updateScene used instead\n" );
			LbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
	}
}

size_t Context::getScenesBuildTempBufferSize(
	const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	const hiprtBuildFlags effectiveBuildFlags = getEffectiveBuildFlags( *this, buildOptions.buildFlags );
	const bool			 useBatchPath		= buildInputs.size() > 1;
	std::vector<hiprtSceneBuildInput> batchInputs;
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( InstanceIDBits < 32 && buildInputs[i].instanceCount >= ( 1u << InstanceIDBits ) )
		{
			std::string msg = Utility::format(
				"The number of instances is %u is greater than the maximum supported number of instances (%u)",
				buildInputs[i].instanceCount,
				1u << InstanceIDBits );
			throw std::runtime_error( msg );
		}

		if ( useBatchPath && batchBuild( buildInputs[i], buildOptions ) ) batchInputs.push_back( buildInputs[i] );
	}

	size_t size = 0;
	if ( !batchInputs.empty() )
	{
		logInfo( "BatchBuild::getSceneBuildTempBufferSize\n" );
		size = BatchBuilder::getTemporaryBufferSize( *this, batchInputs, buildOptions );
	}

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( !( useBatchPath && batchBuild( buildInputs[i], buildOptions ) ) )
		{

			if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
			{
				logInfo( "CustomBvhImport::getSceneBuildTempBufferSize\n" );
				size += BvhImporter::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
			{
				logInfo( "FastBuild::getSceneBuildTempBufferSize\n" );
				size += LbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
			{
				logInfo( "HighQualityBuild::getSceneBuildTempBufferSize\n" );
				size += SbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
			else if ( ( effectiveBuildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
			{
				logInfo( "BalancedBuild::getSceneBuildTempBufferSize\n" );
				size += PlocBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
			else
			{
				logWarn( "Unknow build option => FastBuild::getSceneBuildTempBufferSize used instead\n" );
				size += LbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
		}
	}

	return size;
}

std::vector<hiprtScene> Context::compactScenes( const std::vector<hiprtScene>& scenesIn, cudaStream_t stream )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );

	size_t				size = 0;
	std::vector<size_t> sizes( scenesIn.size() );
	for ( size_t i = 0; i < scenesIn.size(); ++i )
	{
		SceneHeader header;
		checkOro( cuMemcpyDtoH( &header, reinterpret_cast<size_t>( scenesIn[i] ), sizeof( SceneHeader ) ) );
		sizes[i] = getSceneStorageBufferSize(
			header.m_primCount,
			header.m_primNodeCount,
			header.m_boxNodeCount,
			getInstanceNodeSize(),
			getBoxNodeSize(),
			header.m_frameCount );
		size += sizes[i];
	}

	cudaDeviceptr buffer;
	checkOro( cudaMalloc( &buffer, size ) );

	std::vector<hiprtScene> scenesOut( scenesIn.size() );
	for ( size_t i = 0; i < scenesIn.size(); ++i )
	{
		SceneHeader header;
		checkOro( cuMemcpyDtoH( &header, reinterpret_cast<size_t>( scenesIn[i] ), sizeof( SceneHeader ) ) );

		scenesOut[i] = reinterpret_cast<hiprtScene>( buffer );
		MemoryArena storageMemoryArena( scenesOut[i], sizes[i], DefaultAlignment );
		storageMemoryArena.allocate<SceneHeader>();
		void*	  boxNodes	= storageMemoryArena.allocate<uint8_t>( getBoxNodeSize() * header.m_boxNodeCount );
		void*	  primNodes = storageMemoryArena.allocate<uint8_t>( getInstanceNodeSize() * header.m_primNodeCount );
		Instance* instances = storageMemoryArena.allocate<Instance>( header.m_primCount );
		Frame*	  frames	= storageMemoryArena.allocate<Frame>( header.m_frameCount );

		checkOro( cuMemcpyDtoDAsync(
			reinterpret_cast<size_t>( boxNodes ),
			reinterpret_cast<size_t>( header.m_boxNodes ),
			getBoxNodeSize() * header.m_boxNodeCount,
			stream ) );

		checkOro( cuMemcpyDtoDAsync(
			reinterpret_cast<size_t>( primNodes ),
			reinterpret_cast<size_t>( header.m_primNodes ),
			getInstanceNodeSize() * header.m_primNodeCount,
			stream ) );

		checkOro( cuMemcpyDtoDAsync(
			reinterpret_cast<size_t>( instances ),
			reinterpret_cast<size_t>( header.m_instances ),
			sizeof( hiprtTransformHeader ) * header.m_primCount,
			stream ) );

		checkOro( cuMemcpyDtoDAsync(
			reinterpret_cast<size_t>( frames ),
			reinterpret_cast<size_t>( header.m_frames ),
			sizeof( Frame ) * header.m_frameCount,
			stream ) );

		header.m_boxNodes  = boxNodes;
		header.m_primNodes = primNodes;
		header.m_instances = instances;
		header.m_frames	   = frames;
		checkOro(
			cuMemcpyHtoDAsync( reinterpret_cast<size_t>( scenesOut[i] ), &header, sizeof( SceneHeader ), stream ) );

		buffer = reinterpret_cast<uint8_t*>( buffer ) + sizes[i];
	}

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	m_poolHeads[{ reinterpret_cast<cudaDeviceptr>( scenesOut.front() ), size }] = static_cast<uint32_t>( scenesOut.size() );

	checkOro( cudaStreamSynchronize( stream ) );
	destroyScenes( scenesOut );

	return scenesOut;
}

hiprtFuncTable Context::createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes )
{
	checkOro( cuCtxSetCurrent( m_ctxt ) );

	uint8_t* ptr = nullptr;
	checkOro( cudaMalloc(
		reinterpret_cast<void **>( &ptr ),
		sizeof( hiprtFuncTableHeader ) + numGeomTypes * numRayTypes * sizeof( hiprtFuncDataSet ) ) );
	checkOro( cuMemsetD8(
		reinterpret_cast<size_t>( ptr ),
		0,
		sizeof( hiprtFuncTableHeader ) + numGeomTypes * numRayTypes * sizeof( hiprtFuncDataSet ) ) );

	hiprtFuncTableHeader header{
		numGeomTypes, numRayTypes, reinterpret_cast<hiprtFuncDataSet*>( ptr + sizeof( hiprtFuncTableHeader ) ) };
	checkOro( cuMemcpyHtoD( reinterpret_cast<size_t>( ptr ), &header, sizeof( hiprtFuncTableHeader ) ) );

	return reinterpret_cast<hiprtFuncTable>( ptr );
}

void Context::setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set )
{
	checkOro( cuCtxSetCurrent( m_ctxt ) );

	hiprtFuncTableHeader header;
	checkOro( cuMemcpyDtoH( &header, static_cast<uintptr_t>(reinterpret_cast<size_t>( funcTable )), sizeof( hiprtFuncTableHeader ) ) );

	uint32_t index = header.numGeomTypes * rayType + geomType;
	checkOro(
		cuMemcpyHtoD( reinterpret_cast<size_t>( &header.funcDataSets[index] ), &set, sizeof( hiprtFuncDataSet ) ) );
}

void Context::destroyFuncTable( hiprtFuncTable funcTable )
{
	checkOro( cuCtxSetCurrent( m_ctxt ) );
	checkOro( cudaFree( reinterpret_cast<void *>( funcTable ) ) );
}

void Context::createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut )
{
	checkOro( cuCtxSetCurrent( m_ctxt ) );

	const size_t stackEntrySize =
		input.entryType == hiprtStackEntryTypeInstance ? sizeof( hiprtInstanceStackEntry ) : sizeof( uint32_t );
	if ( input.type == hiprtStackTypeDynamic )
	{
		cudaDeviceProp prop;
		checkOro( cudaGetDeviceProperties( &prop, m_device ) );
		const uint32_t maxThreadsPerMultiProcessor =
			prop.maxThreadsPerMultiProcessor <= 0 ? 2048u : prop.maxThreadsPerMultiProcessor;
		const uint32_t		   stackCount  = prop.multiProcessorCount * maxThreadsPerMultiProcessor;
		const uint32_t		   activeWarps = stackCount / prop.warpSize;
		size_t				   size		   = activeWarps * sizeof( uint32_t ) + stackCount * input.stackSize * stackEntrySize;
		hiprtGlobalStackBuffer stackBuffer{ input.stackSize, stackCount, nullptr };
		checkOro( cudaMalloc( reinterpret_cast<void **>( &stackBuffer.stackData ), size ) );
		// checkOro( cuMemsetD8( reinterpret_cast<size_t>( stackBuffer.stackData ), 0, sizeof( uint32_t ) * stackCount ) );
		stackBufferOut = stackBuffer;
	}
	else
	{
		const uint32_t		   stackStride = getWarpSize();
		size_t				   size		   = input.stackSize * input.threadCount * stackStride * stackEntrySize;
		hiprtGlobalStackBuffer stackBuffer{ input.stackSize, input.threadCount, nullptr };
		checkOro( cudaMalloc( reinterpret_cast<void **>( &stackBuffer.stackData ), size ) );
		stackBufferOut = stackBuffer;
	}
}

void Context::destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer )
{
	checkOro( cuCtxSetCurrent( m_ctxt ) );
	checkOro( cudaFree( reinterpret_cast<void *>( stackBuffer.stackData ) ) );
}

void Context::saveGeometry( hiprtGeometry inGeometry, const std::string& filename )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );

	size_t size = 0;
	{
		GeomHeader header;
		checkOro( cuMemcpyDtoH( &header, reinterpret_cast<size_t>( inGeometry ), sizeof( GeomHeader ) ) );
		size = header.m_size;
	}

	std::vector<uint8_t> buffer( size );
	checkOro( cuMemcpyDtoH( buffer.data(), reinterpret_cast<size_t>( inGeometry ), size ) );

	GeomHeader header;
	std::memcpy( &header, buffer.data(), sizeof( GeomHeader ) );
	std::uintptr_t offset = reinterpret_cast<std::uintptr_t>( inGeometry );
	header.m_boxNodes	  = reinterpret_cast<void*>( reinterpret_cast<std::uintptr_t>( header.m_boxNodes ) - offset );
	header.m_primNodes	  = reinterpret_cast<void*>( reinterpret_cast<std::uintptr_t>( header.m_primNodes ) - offset );
	std::memcpy( buffer.data(), &header, sizeof( GeomHeader ) );

	std::ofstream file( filename, std::ios::out | std::ios::binary );
	file.write( reinterpret_cast<char*>( buffer.data() ), header.m_size );
}

hiprtGeometry Context::loadGeometry( const std::string& filename )
{
	std::ifstream file( filename, std::ios::in | std::ios::binary );

	size_t size = 0;
	{
		GeomHeader header;
		file.read( reinterpret_cast<char*>( &header ), sizeof( GeomHeader ) );
		size = header.m_size;
		if ( header.m_rtip != getRtip() && ( getRtip() >= 31 || header.m_rtip >= 31 ) )
		{
			std::string msg = Utility::format(
				"RTIP of the loaded geometry (%u) is not compatible with the RTIP of the current context (%u).",
				header.m_rtip,
				getRtip() );
			throw std::runtime_error( msg );
		}
	}

	std::vector<uint8_t> buffer( size );
	file.clear();
	file.seekg( 0, std::ios::beg );
	file.read( reinterpret_cast<char*>( buffer.data() ), size );

	hiprtGeometry geometry;
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	checkOro( cudaMalloc( reinterpret_cast<void **>( &geometry ), size ) );

	GeomHeader header;
	std::memcpy( &header, buffer.data(), sizeof( GeomHeader ) );
	std::uintptr_t offset = reinterpret_cast<std::uintptr_t>( geometry );
	header.m_boxNodes	  = reinterpret_cast<void*>( reinterpret_cast<std::uintptr_t>( header.m_boxNodes ) + offset );
	header.m_primNodes	  = reinterpret_cast<void*>( reinterpret_cast<std::uintptr_t>( header.m_primNodes ) + offset );
	std::memcpy( buffer.data(), &header, sizeof( GeomHeader ) );

	checkOro( cuMemcpyHtoD( reinterpret_cast<size_t>( geometry ), buffer.data(), header.m_size ) );

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	m_poolHeads[{ reinterpret_cast<cudaDeviceptr>( geometry ), header.m_size }] = 1u;

	return geometry;
}

void Context::saveScene( [[maybe_unused]] hiprtScene inScene, [[maybe_unused]] const std::string& filename )
{
	throw std::runtime_error( "Not implemented" );
}

hiprtScene Context::loadScene( [[maybe_unused]] const std::string& filename ) { throw std::runtime_error( "Not implemented" ); }

void Context::exportGeometryAabb( hiprtGeometry inGeometry, float3& outAabbMin, float3& outAabbMax )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );

	GeomHeader header;
	checkOro( cuMemcpyDtoH( &header, reinterpret_cast<size_t>( inGeometry ), sizeof( GeomHeader ) ) );

	constexpr uint32_t Alignment = alignof( Box8Node ) > alignof( Box4Node ) ? alignof( Box8Node ) : alignof( Box4Node );
	constexpr uint32_t Size		 = sizeof( Box8Node ) > sizeof( Box4Node ) ? sizeof( Box8Node ) : sizeof( Box4Node );
	alignas( Alignment ) uint8_t root[Size];
	checkOro( cuMemcpyDtoH( root, reinterpret_cast<size_t>( header.m_boxNodes ), getBoxNodeSize() ) );

	Aabb box   = getRtip() >= 31 ? reinterpret_cast<Box8Node*>( root )->aabb() : reinterpret_cast<Box4Node*>( root )->aabb();
	outAabbMin = box.m_min;
	outAabbMax = box.m_max;
}

void Context::exportSceneAabb( hiprtScene inScene, float3& outAabbMin, float3& outAabbMax )
{
	// checkOro( cuCtxSetCurrent( m_ctxt ) );

	SceneHeader header;
	checkOro( cuMemcpyDtoH( &header, reinterpret_cast<size_t>( inScene ), sizeof( SceneHeader ) ) );

	constexpr uint32_t Alignment = alignof( Box8Node ) > alignof( Box4Node ) ? alignof( Box8Node ) : alignof( Box4Node );
	constexpr uint32_t Size		 = sizeof( Box8Node ) > sizeof( Box4Node ) ? sizeof( Box8Node ) : sizeof( Box4Node );
	alignas( Alignment ) uint8_t root[Size];
	checkOro( cuMemcpyDtoH( root, reinterpret_cast<size_t>( header.m_boxNodes ), getBoxNodeSize() ) );

	Aabb box   = getRtip() >= 31 ? reinterpret_cast<Box8Node*>( root )->aabb() : reinterpret_cast<Box4Node*>( root )->aabb();
	outAabbMin = box.m_min;
	outAabbMax = box.m_max;
}

void Context::buildKernels(
	const std::vector<const char*>&		 funcNames,
	const std::string&					 src,
	const std::filesystem::path&		 moduleName,
	std::vector<const char*>&			 headers,
	std::vector<const char*>&			 includeNames,
	std::vector<const char*>&			 options,
	uint32_t							 numGeomTypes,
	uint32_t							 numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets,
	std::vector<CUfunction>&			 functions,
	CUmodule&							 module,
	bool								 cache )
{
	checkOro( cuCtxSetCurrent( m_ctxt ) );
	m_compiler.buildKernels(
		*this,
		funcNames,
		src,
		moduleName,
		headers,
		includeNames,
		options,
		numGeomTypes,
		numRayTypes,
		funcNameSets,
		functions,
		module,
		true,
		cache );
}

void Context::setCacheDir( const std::filesystem::path& path ) { m_compiler.setCacheDir( path ); }

uint32_t Context::getSMCount() const
{
	int smCount;
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	checkOro( cudaDeviceGetAttribute( &smCount, cudaDevAttrMultiProcessorCount, m_device ) );
	return smCount;
}

uint32_t Context::getMaxBlockSize() const
{
	cudaDeviceProp prop;
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	checkOro( cudaGetDeviceProperties( &prop, m_device ) );
	return prop.maxThreadsPerBlock;
}

uint32_t Context::getMaxGridSize() const
{
	cudaDeviceProp prop;
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	checkOro( cudaGetDeviceProperties( &prop, m_device ) );
	return prop.maxGridSize[0];
}

std::string Context::getDeviceName() const
{
	cudaDeviceProp prop;
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	checkOro( cudaGetDeviceProperties( &prop, m_device ) );
	return std::string( prop.name );
}

// std::string Context::getGcnArchName() const
// {
// 	cudaDeviceProp prop;
// 	checkOro( cuCtxSetCurrent( m_ctxt ) );
// 	checkOro( cudaGetDeviceProperties( &prop, m_device ) );
// 	return std::string( prop.gcnArchName );
// }

std::string Context::getDriverVersion() const
{
	int driverVersion;
	// checkOro( cuCtxSetCurrent( m_ctxt ) );
	checkOro( cudaDriverGetVersion( &driverVersion ) );
	return std::to_string( driverVersion );
}

uint32_t Context::getRtip() const
{
	return 0;
}

uint32_t Context::getBranchingFactor() const
{
	if ( getRtip() >= 31 ) return 8;
	return 4;
}

uint32_t Context::getWarpSize() const
{
	std::string deviceName = getDeviceName();
	std::string archName   = "getGcnArchName()";

	uint32_t archNumber = 0;
	if ( archName.substr( 0, 3 ) == "gfx" )
	{
		std::string numberPart = archName.substr( 3 );
		archNumber			   = std::stoi( numberPart );
	}

	uint32_t warpSize = 32;
	if ( deviceName.find( "NVIDIA" ) == std::string::npos )
	{
		if ( archNumber < 1030 ) warpSize = 64;
	}

	return warpSize;
}

size_t Context::getTriangleNodeSize() const
{
	if ( getRtip() >= 31 ) return sizeof( TrianglePacketNode );
	return sizeof( TrianglePairNode );
}

size_t Context::getBoxNodeSize() const
{
	if ( getRtip() >= 31 ) return sizeof( Box8Node );
	return sizeof( Box4Node );
}

size_t Context::getInstanceNodeSize() const
{
	if ( getRtip() >= 31 ) return sizeof( HwInstanceNode );
	return sizeof( UserInstanceNode );
}
} // namespace hiprt
