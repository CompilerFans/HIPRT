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

#include <hiprt/impl/AabbList.h>
#include <hiprt/impl/BvhCommon.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/PlocBuilder.h>
#include <hiprt/impl/TriangleMesh.h>

namespace hiprt
{
size_t PlocBuilder::getTemporaryBufferSize( const size_t count )
{
	return RoundUp( sizeof( Aabb ), DefaultAlignment ) + 4 * RoundUp( sizeof( uint32_t ) * count, DefaultAlignment ) +
		   RoundUp( count * sizeof( ScratchNode ), DefaultAlignment ) +
		   RoundUp( count * sizeof( ReferenceNode ), DefaultAlignment ) + RoundUp( sizeof( uint32_t ), DefaultAlignment );
}

size_t PlocBuilder::getTemporaryBufferSize(
	[[maybe_unused]] Context& context, const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t primCount = getPrimCount( buildInput );
	size_t		 size	   = getTemporaryBufferSize( primCount );
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh )
	{
		const hiprtTriangleMeshPrimitive& mesh	   = buildInput.primitive.triangleMesh;
		const bool						  pairable = mesh.triangleCount > 2 && mesh.trianglePairCount == 0;
		const bool pairTriangles = pairable && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableTrianglePairing );
		if ( pairTriangles ) size += RoundUp( sizeof( uint2 ) * primCount, DefaultAlignment );
	}
	return size;
}

size_t PlocBuilder::getTemporaryBufferSize(
	[[maybe_unused]] Context&				 context,
	const hiprtSceneBuildInput&				 buildInput,
	[[maybe_unused]] const hiprtBuildOptions buildOptions )
{
	return getTemporaryBufferSize( buildInput.instanceCount );
}

size_t PlocBuilder::getStorageBufferSize(
	Context& context, const hiprtGeometryBuildInput& buildInput, [[maybe_unused]] const hiprtBuildOptions buildOptions )
{
	const size_t primCount	   = getPrimCount( buildInput );
	const size_t primNodeCount = getMaxPrimNodeCount( buildInput, context.getRtip(), primCount );
	const size_t primNodeSize  = getPrimNodeSize( buildInput, context.getTriangleNodeSize() );
	const size_t boxNodeCount  = getMaxBoxNodeCount( buildInput, context.getRtip(), primCount );
	return getGeometryStorageBufferSize( primNodeCount, boxNodeCount, primNodeSize, context.getBoxNodeSize() );
}

size_t PlocBuilder::getStorageBufferSize(
	Context& context, const hiprtSceneBuildInput& buildInput, [[maybe_unused]] const hiprtBuildOptions buildOptions )
{
	const size_t primCount	  = buildInput.instanceCount;
	const size_t boxNodeCount = getMaxBoxNodeCount( buildInput, context.getRtip(), primCount );
	return getSceneStorageBufferSize(
		primCount, primCount, boxNodeCount, context.getInstanceNodeSize(), context.getBoxNodeSize(), buildInput.frameCount );
}

void PlocBuilder::build(
	Context&					   context,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	oroStream					   stream,
	hiprtDevicePtr				   buffer )
{
	const size_t storageSize = getStorageBufferSize( context, buildInput, buildOptions );
	const size_t tempSize	 = getTemporaryBufferSize( context, buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );
	MemoryArena	 temporaryMemoryArena( temporaryBuffer, tempSize, DefaultAlignment );

	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		TriangleMesh mesh( buildInput.primitive.triangleMesh );
		if ( context.getRtip() >= 31 )
			build<Box8Node, TrianglePacketNode>(
				context, mesh, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		else
			build<Box4Node, TrianglePairNode>(
				context, mesh, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		AabbList list( buildInput.primitive.aabbList );
		if ( context.getRtip() >= 31 )
			build<Box8Node, CustomNode>(
				context, list, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		else
			build<Box4Node, CustomNode>(
				context, list, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void PlocBuilder::build(
	Context&					context,
	const hiprtSceneBuildInput& buildInput,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	oroStream					stream,
	hiprtDevicePtr				buffer )
{
	const size_t storageSize = getStorageBufferSize( context, buildInput, buildOptions );
	const size_t tempSize	 = getTemporaryBufferSize( context, buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );
	MemoryArena	 temporaryMemoryArena( temporaryBuffer, tempSize, DefaultAlignment );

	switch ( buildInput.frameType )
	{
	case hiprtFrameTypeSRT: {
		InstanceList<SRTFrame> list( buildInput );
		if ( context.getRtip() >= 31 )
			build<Box8Node, HwInstanceNode>(
				context, list, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		else
			build<Box4Node, UserInstanceNode>(
				context, list, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	case hiprtFrameTypeMatrix: {
		InstanceList<MatrixFrame> list( buildInput );
		if ( context.getRtip() >= 31 )
			build<Box8Node, HwInstanceNode>(
				context, list, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		else
			build<Box4Node, UserInstanceNode>(
				context, list, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void PlocBuilder::update(
	Context&						context,
	const hiprtGeometryBuildInput&	buildInput,
	const hiprtBuildOptions			buildOptions,
	[[maybe_unused]] hiprtDevicePtr temporaryBuffer,
	oroStream						stream,
	hiprtDevicePtr					buffer )
{
	const size_t storageSize = getStorageBufferSize( context, buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );

	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		TriangleMesh mesh( buildInput.primitive.triangleMesh );
		if ( context.getRtip() >= 31 )
			update<Box8Node, TrianglePacketNode>( context, mesh, buildOptions, stream, storageMemoryArena );
		else
			update<Box4Node, TrianglePairNode>( context, mesh, buildOptions, stream, storageMemoryArena );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		AabbList list( buildInput.primitive.aabbList );
		if ( context.getRtip() >= 31 )
			update<Box8Node, CustomNode>( context, list, buildOptions, stream, storageMemoryArena );
		else
			update<Box4Node, CustomNode>( context, list, buildOptions, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void PlocBuilder::update(
	Context&						context,
	const hiprtSceneBuildInput&		buildInput,
	const hiprtBuildOptions			buildOptions,
	[[maybe_unused]] hiprtDevicePtr temporaryBuffer,
	oroStream						stream,
	hiprtDevicePtr					buffer )
{
	const size_t storageSize = getStorageBufferSize( context, buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );

	switch ( buildInput.frameType )
	{
	case hiprtFrameTypeSRT: {
		InstanceList<SRTFrame> list( buildInput );
		if ( context.getRtip() >= 31 )
			update<Box8Node, HwInstanceNode>( context, list, buildOptions, stream, storageMemoryArena );
		else
			update<Box4Node, UserInstanceNode>( context, list, buildOptions, stream, storageMemoryArena );
		break;
	}
	case hiprtFrameTypeMatrix: {
		InstanceList<MatrixFrame> list( buildInput );
		if ( context.getRtip() >= 31 )
			update<Box8Node, HwInstanceNode>( context, list, buildOptions, stream, storageMemoryArena );
		else
			update<Box4Node, UserInstanceNode>( context, list, buildOptions, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}
} // namespace hiprt
