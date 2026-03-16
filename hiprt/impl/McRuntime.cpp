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
} // namespace hiprt::mc
