//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <filesystem>
#include <string_view>

#include <cuda.h>

namespace hiprt::mc
{
enum class LinkInputType
{
	Ptx,
	Cubin,
	Fatbinary,
};

using Module = CUmodule;
using Function = CUfunction;
using LinkState = CUlinkState;

LinkState createLinkState( unsigned int numOptions, CUjit_option* options, void** optionValues );
void destroyLinkState( LinkState state );
void addFile( LinkState state, LinkInputType type, const std::filesystem::path& path );
void addData( LinkState state, LinkInputType type, std::string_view data, const char* name );
void completeLink( LinkState state, void** imageOut, size_t* sizeOut );

Module loadModule( const void* image );
Function getFunction( Module module, const char* name );
void unloadModule( Module module );
void launchKernel(
	Function function,
	uint32_t gx,
	uint32_t gy,
	uint32_t gz,
	uint32_t bx,
	uint32_t by,
	uint32_t bz,
	uint32_t sharedMemBytes,
	cudaStream_t stream,
	void** args );
void occupancyMaxPotentialBlockSize( int* minGridSize, int* blockSize, Function function );
int getFunctionAttribute( Function function, CUfunction_attribute attribute );
} // namespace hiprt::mc
