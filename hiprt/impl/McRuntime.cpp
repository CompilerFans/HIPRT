//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <hiprt/impl/Error.h>
#include <hiprt/impl/McRuntime.h>

namespace hiprt::mc
{
namespace
{
CUjitInputType toCuInputType( LinkInputType type )
{
	switch ( type )
	{
	case LinkInputType::Ptx: return CU_JIT_INPUT_PTX;
	case LinkInputType::Cubin: return CU_JIT_INPUT_CUBIN;
	case LinkInputType::Fatbinary: return CU_JIT_INPUT_FATBINARY;
	}
	return CU_JIT_INPUT_PTX;
}
} // namespace

LinkState createLinkState( unsigned int numOptions, CUjit_option* options, void** optionValues )
{
	LinkState state = nullptr;
	checkOro( cuLinkCreate( numOptions, options, optionValues, &state ) );
	return state;
}

void destroyLinkState( LinkState state )
{
	if ( state != nullptr ) checkOro( cuLinkDestroy( state ) );
}

void addFile( LinkState state, LinkInputType type, const std::filesystem::path& path )
{
	checkOro( cuLinkAddFile( state, toCuInputType( type ), const_cast<char*>( path.string().c_str() ), 0, nullptr, nullptr ) );
}

void addData( LinkState state, LinkInputType type, std::string_view data, const char* name )
{
	checkOro( cuLinkAddData(
		state,
		toCuInputType( type ),
		const_cast<char*>( data.data() ),
		data.size(),
		const_cast<char*>( name ),
		0,
		nullptr,
		nullptr ) );
}

void completeLink( LinkState state, void** imageOut, size_t* sizeOut ) { checkOro( cuLinkComplete( state, imageOut, sizeOut ) ); }

Module loadModule( const void* image )
{
	Module module = nullptr;
	checkOro( cuModuleLoadData( &module, image ) );
	return module;
}

Function getFunction( Module module, const char* name )
{
	Function function = nullptr;
	checkOro( cuModuleGetFunction( &function, module, name ) );
	return function;
}

void unloadModule( Module module )
{
	if ( module != nullptr ) checkOro( cuModuleUnload( module ) );
}

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
	void** args )
{
	checkOro( cuLaunchKernel( function, gx, gy, gz, bx, by, bz, sharedMemBytes, stream, args, nullptr ) );
}

void occupancyMaxPotentialBlockSize( int* minGridSize, int* blockSize, Function function )
{
	checkOro( cuOccupancyMaxPotentialBlockSize( minGridSize, blockSize, function, 0, 0, 0 ) );
}

int getFunctionAttribute( Function function, CUfunction_attribute attribute )
{
	int value = 0;
	checkOro( cuFuncGetAttribute( &value, attribute, function ) );
	return value;
}

void init() { checkOro( cuInit( 0 ) ); }

CUdevice getDevice( int deviceOrdinal )
{
	CUdevice device = 0;
	checkOro( cuDeviceGet( &device, deviceOrdinal ) );
	return device;
}

CUcontext retainPrimaryContext( CUdevice device )
{
	CUcontext context = nullptr;
	checkOro( cuDevicePrimaryCtxRetain( &context, device ) );
	return context;
}

void setCurrentContext( CUcontext context ) { checkOro( cuCtxSetCurrent( context ) ); }

void releasePrimaryContext( CUdevice device ) { checkOro( cuDevicePrimaryCtxRelease( device ) ); }

void memsetD8( CUdeviceptr dst, unsigned char value, size_t byteCount ) { checkOro( cuMemsetD8( dst, value, byteCount ) ); }

void memcpyHtoD( CUdeviceptr dst, const void* src, size_t byteCount ) { checkOro( cuMemcpyHtoD( dst, src, byteCount ) ); }

void memcpyDtoH( void* dst, CUdeviceptr src, size_t byteCount ) { checkOro( cuMemcpyDtoH( dst, src, byteCount ) ); }
} // namespace hiprt::mc
