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

#include <array>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>

#include <hiprt/hiprt.h>
#include <hiprt/impl/Compiler.h>
#include <hiprt/impl/Context.h>
#include <hiprt/impl/Error.h>
#include <hiprt/impl/Utility.h>

namespace hiprt
{
namespace
{
bool getRuntimeKernelDiskCacheEnabled()
{
	const std::string disableCache = Utility::getEnvVariable( "HIPRT_DISABLE_RUNTIME_KERNEL_CACHE" );
	if ( !disableCache.empty() && disableCache != "0" && disableCache != "false" && disableCache != "FALSE" )
		return false;

#if defined( HIPRT_RUNTIME_KERNEL_CACHE_DEFAULT )
	return HIPRT_RUNTIME_KERNEL_CACHE_DEFAULT != 0;
#else
	return true;
#endif
}

constexpr auto LinkLogSize = 8192u;

std::string getNvrtcArchOpt( Context& context )
{
	int major = 0;
	int minor = 0;
	checkOro( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, context.getDevice() ) );
	checkOro( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, context.getDevice() ) );
	return "--gpu-architecture=compute_" + std::to_string( major ) + std::to_string( minor );
}

bool isElfBinary( const std::string_view binary )
{
	return binary.size() >= 4 && static_cast<unsigned char>( binary[0] ) == 0x7F && binary[1] == 'E' && binary[2] == 'L' &&
		   binary[3] == 'F';
}

bool isFatbinBinary( const std::string_view binary )
{
	constexpr std::string_view ClangOffloadMagic = "__CLANG_OFFLOAD_";
	return binary.size() >= ClangOffloadMagic.size() && binary.substr( 0, ClangOffloadMagic.size() ) == ClangOffloadMagic;
}

std::string getNvrtcCompiledBinary( nvrtcProgram prog )
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

std::string quoteShellArg( const std::string& arg )
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

std::string findCudaCompiler()
{
	const std::string preferredExternal = Utility::getEnvVariable( "HIPRT_EXTERNAL_DEVICE_COMPILER" );
	const bool		  preferMxcc		  = preferredExternal == "mxcc";
	const bool		  preferCucc		  = preferredExternal == "cucc";
	const bool		  preferNvcc		  = preferredExternal == "nvcc";

	const std::vector<std::string> candidates = {
		preferMxcc ? "/opt/maca/mxgpu_llvm/bin/mxcc" : std::string(),
		preferCucc ? "/opt/maca/tools/cu-bridge/bin/cucc" : std::string(),
		preferNvcc ? "nvcc" : std::string(),
		Utility::getEnvVariable( "HIPRT_CUDA_COMPILER" ),
		Utility::getEnvVariable( "CUBIN_COMPILER" ),
		Utility::getEnvVariable( "CUDACXX" ),
		"/opt/maca/mxgpu_llvm/bin/mxcc",
		Utility::getEnvVariable( "CUDA_PATH" ).empty() ? std::string() : Utility::getEnvVariable( "CUDA_PATH" ) + "/bin/nvcc",
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

	throw std::runtime_error( "Unable to locate a CUDA-compatible compiler for bitcode fallback compilation." );
}

std::string compileSourceToCubin(
	Context& context, const std::string& source, const std::vector<std::string>& extraOptions, const std::string& stem )
{
	const std::filesystem::path tempDir =
		std::filesystem::temp_directory_path() /
		Utility::format( "hiprt-bitcode-%08x", Utility::hashString( source + stem + std::to_string( std::rand() ) ) );
	std::filesystem::create_directories( tempDir );

	const std::filesystem::path srcPath = tempDir / ( stem + ".cu" );

	{
		std::ofstream file( srcPath, std::ios::out | std::ios::binary );
		file.write( source.data(), static_cast<std::streamsize>( source.size() ) );
	}

	const std::string compiler = findCudaCompiler();
	std::ostringstream cmd;
	const bool useMxcc = std::filesystem::path( compiler ).filename() == "mxcc";
	const std::filesystem::path outPath = tempDir / ( stem + ( useMxcc ? ".fatbin" : ".cubin" ) );
	if ( useMxcc )
	{
		const std::string macaPath = Utility::getEnvVariable( "MACA_PATH" ).empty() ? "/opt/maca" : Utility::getEnvVariable( "MACA_PATH" );
		const std::string cudaPath =
			Utility::getEnvVariable( "CUDA_PATH" ).empty() ? macaPath + "/tools/cu-bridge" : Utility::getEnvVariable( "CUDA_PATH" );
		const std::string offloadArch =
			Utility::getEnvVariable( "MXCC_OFFLOAD_ARCH" ).empty() ? "xcore1000" : Utility::getEnvVariable( "MXCC_OFFLOAD_ARCH" );

		cmd << quoteShellArg( compiler ) << " -O3 -std=c++17 -fatbin -use-fast-math"
			<< " -x maca -fgpu-rdc --include cuda_runtime.h -D__CUDACC__"
			<< " -I" << quoteShellArg( Utility::getRootDir().string() )
			<< " -I" << quoteShellArg( ( Utility::getRootDir() / "contrib/Orochi" ).string() )
			<< " -I" << quoteShellArg( cudaPath + "/include" )
			<< " -I" << quoteShellArg( macaPath + "/include" )
			<< " --offload-arch=" << offloadArch
			<< " " << quoteShellArg( srcPath.string() );
	}
	else
	{
		cmd << quoteShellArg( compiler ) << " -x cu " << quoteShellArg( srcPath.string() )
			<< " -O3 -std=c++17 --device-c -cubin --use_fast_math"
			<< " -I" << quoteShellArg( Utility::getRootDir().string() )
			<< " -I" << quoteShellArg( ( Utility::getRootDir() / "contrib/Orochi" ).string() );
	}
	for ( const auto& option : extraOptions )
		cmd << " " << option;
	cmd << " -o " << quoteShellArg( outPath.string() );

	if ( std::system( cmd.str().c_str() ) != 0 )
		throw std::runtime_error( "External compiler fallback failed for bitcode source: " + srcPath.string() );

	std::ifstream file( outPath, std::ios::in | std::ios::binary | std::ios::ate );
	if ( !file.is_open() ) throw std::runtime_error( "Failed to open fallback cubin: " + outPath.string() );

	const size_t size = static_cast<size_t>( file.tellg() );
	file.seekg( 0, std::ios::beg );
	std::string cubin( size, '\0' );
	file.read( cubin.data(), static_cast<std::streamsize>( size ) );
	return cubin;
}
} // namespace

Compiler::~Compiler()
{
	clear();
}

void Compiler::clear()
{
	for ( auto& module : m_moduleCache )
		checkOro( cuModuleUnload( module.second ) );
	m_moduleCache.clear();
	m_kernelCache.clear();
}

Kernel Compiler::getKernel(
	Context&					 context,
	const std::filesystem::path& moduleName,
	const std::string&			 funcName,
	std::vector<const char*>&	 options,
	uint32_t					 numHeaders,
	const char**				 headersIn,
	const char**				 includeNamesIn )
{
	std::lock_guard<std::mutex> lock( m_kernelMutex );

	const std::string cacheName = moduleName.string() + funcName;
	auto			  cacheEntry = m_kernelCache.find( cacheName );
	if ( cacheEntry != m_kernelCache.end() ) return cacheEntry->second;

	std::vector<const char*>	  funcNames = { funcName.c_str() };
	std::vector<const char*>	  headers;
	std::vector<const char*>	  includeNames;
	std::vector<hiprtFuncNameSet> funcNameSets;
	std::vector<CUfunction>		  functions;
	CUmodule					  module = nullptr;

	if ( numHeaders == 0 )
	{
		std::string src = readSourceCode( moduleName );
		buildKernels(
			context, funcNames, src, moduleName, headers, includeNames, options, 0, 0, funcNameSets, functions, module, false, true );
	}
	else
	{
		std::vector<std::string> headerData( numHeaders - 1 );
		for ( uint32_t i = 0; i < numHeaders - 1; ++i )
		{
			includeNames.push_back( includeNamesIn[i] );
			headerData[i] = headersIn[i];
			headers.push_back( headerData[i].c_str() );
		}

		const std::string src = headersIn[numHeaders - 1];
		buildKernels(
			context, funcNames, src, moduleName, headers, includeNames, options, 0, 0, funcNameSets, functions, module, false, true );
	}

	Kernel kernel( functions.back() );
	m_kernelCache[cacheName] = kernel;
	return kernel;
}

void Compiler::buildProgram(
	const std::vector<const char*>& funcNames,
	const std::string&				src,
	const std::filesystem::path&	moduleName,
	std::vector<const char*>&		headers,
	std::vector<const char*>&		includeNames,
	std::vector<const char*>&		options,
	nvrtcProgram&					progOut )
{
	checkOrortc( nvrtcCreateProgram(
		&progOut,
		src.c_str(),
		moduleName.string().c_str(),
		static_cast<int>( headers.size() ),
		headers.data(),
		includeNames.data() ) );

#if !defined( HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND ) || HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND == 0
	for ( const char* funcName : funcNames )
		checkOrortc( nvrtcAddNameExpression( progOut, funcName ) );
#endif

	const nvrtcResult result = nvrtcCompileProgram( progOut, static_cast<int>( options.size() ), options.data() );
	if ( result != NVRTC_SUCCESS )
	{
		size_t logSize = 0;
		checkOrortc( nvrtcGetProgramLogSize( progOut, &logSize ) );
		if ( logSize == 0 ) throw std::runtime_error( "Runtime compilation failed with an empty NVRTC log." );

		std::string log( logSize, '\0' );
		checkOrortc( nvrtcGetProgramLog( progOut, &log[0] ) );
		throw std::runtime_error( "Runtime compilation failed:\n" + log );
	}
}

void Compiler::buildKernels(
	Context&							 context,
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
	bool								 extended,
	bool								 cache )
{
	const bool useDiskCache = cache && getRuntimeKernelDiskCacheEnabled();
	if ( useDiskCache && !std::filesystem::exists( m_cacheDirectory ) && !std::filesystem::create_directory( m_cacheDirectory ) )
		throw std::runtime_error( "Cannot create cache directory" );

	std::lock_guard<std::mutex> lock( m_moduleMutex );
	auto						cacheEntry = m_moduleCache.find( moduleName.string() );
	if ( cacheEntry != m_moduleCache.end() )
	{
		module = cacheEntry->second;
	}
	else
	{
		const std::string cacheName = getCacheFilename( context, src, moduleName, options, funcNameSets, numGeomTypes, numRayTypes );
		const bool		  upToDate  = isCachedFileUpToDate( m_cacheDirectory / cacheName, moduleName );

		nvrtcProgram prog = nullptr;
		std::string	 binary;
		if ( upToDate && useDiskCache )
		{
			binary = loadCacheFileToBinary( cacheName );
		}
		else
		{
			std::string extSrc = src;
			if ( extended )
			{
				extSrc = "#include <hiprt/impl/hiprt_device_impl.h>\n";
				addCustomFuncsSwitchCase( extSrc, funcNameSets, numGeomTypes, numRayTypes );
				extSrc += "\n" + src;
			}

			std::vector<const char*> opts = options;
			std::string				 includePath = "-I" + Utility::getRootDir().string();
			opts.push_back( includePath.c_str() );
			addCommonOpts( context, opts, extended );

			buildProgram( funcNames, extSrc, moduleName, headers, includeNames, opts, prog );

			binary = getNvrtcCompiledBinary( prog );
			if ( binary.empty() )
				throw std::runtime_error( "Runtime compilation succeeded but emitted neither PTX nor CUBIN." );
			checkOrortc( nvrtcDestroyProgram( &prog ) );

			if ( useDiskCache ) cacheBinaryToFile( binary, cacheName );
		}

		checkOro( cuModuleLoadData( &module, binary.data() ) );
		m_moduleCache[moduleName.string()] = module;
	}

	for ( const char* funcName : funcNames )
	{
		CUfunction func = nullptr;
		checkOro( cuModuleGetFunction( &func, module, funcName ) );
		functions.push_back( func );
	}
}

void Compiler::buildKernelsFromBitcode(
	Context&							 context,
	const std::vector<const char*>&		 funcNames,
	const std::filesystem::path&		 moduleName,
	const std::string_view				 bitcodeBinary,
	uint32_t							 numGeomTypes,
	uint32_t							 numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets,
	std::vector<CUfunction>&			 functions,
	bool								 cache )
{
	const bool useDiskCache = cache && getRuntimeKernelDiskCacheEnabled();
	if ( useDiskCache && !std::filesystem::exists( m_cacheDirectory ) && !std::filesystem::create_directory( m_cacheDirectory ) )
		throw std::runtime_error( "Cannot create cache directory" );

	const std::string binaryKey( bitcodeBinary.data(), bitcodeBinary.size() );
	const std::string cacheKey = "bitcode:" + moduleName.string() + ":" +
								 Utility::format( "%08x", Utility::hashString( binaryKey + std::to_string( numGeomTypes ) +
																			  std::to_string( numRayTypes ) ) );

	std::lock_guard<std::mutex> lock( m_moduleMutex );
	auto						cacheEntry = m_moduleCache.find( cacheKey );
	CUmodule					module		= nullptr;
	if ( cacheEntry != m_moduleCache.end() )
	{
		module = cacheEntry->second;
	}
	else
	{
		const std::string diskCacheName =
			getCacheFilename( context, binaryKey, moduleName, std::nullopt, funcNameSets, numGeomTypes, numRayTypes );
		const bool upToDate = isCachedFileUpToDate( m_cacheDirectory / diskCacheName, moduleName );

		std::string binary;
		if ( upToDate && useDiskCache )
		{
			binary = loadCacheFileToBinary( diskCacheName );
		}
		else
		{
			const std::string customFuncBitcodeBinary =
				buildFunctionTableBitcode( context, numGeomTypes, numRayTypes, funcNameSets );
				const std::filesystem::path bcPath = getBitcodePath();
				const CUjitInputType userBinaryType = isElfBinary( bitcodeBinary )
														 ? CU_JIT_INPUT_CUBIN
														 : ( isFatbinBinary( bitcodeBinary ) ? CU_JIT_INPUT_FATBINARY : CU_JIT_INPUT_PTX );
				const CUjitInputType customBinaryType = isElfBinary( customFuncBitcodeBinary )
														   ? CU_JIT_INPUT_CUBIN
														   : ( isFatbinBinary( customFuncBitcodeBinary ) ? CU_JIT_INPUT_FATBINARY
																										 : CU_JIT_INPUT_PTX );

			std::array<char, LinkLogSize> errorLog{};
			std::array<char, LinkLogSize> infoLog{};
			float						  wallTime = 0.0f;

			CUjit_option options[] = {
				CU_JIT_WALL_TIME,
				CU_JIT_INFO_LOG_BUFFER,
				CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
				CU_JIT_ERROR_LOG_BUFFER,
				CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
				CU_JIT_LOG_VERBOSE,
			};
			void* optionValues[] = {
				&wallTime,
				infoLog.data(),
				reinterpret_cast<void*>( static_cast<uintptr_t>( infoLog.size() ) ),
				errorLog.data(),
				reinterpret_cast<void*>( static_cast<uintptr_t>( errorLog.size() ) ),
				reinterpret_cast<void*>( static_cast<uintptr_t>( 1 ) ),
			};

			CUlinkState linkState = nullptr;
			checkOro( cuLinkCreate(
				static_cast<unsigned int>( sizeof( options ) / sizeof( options[0] ) ), options, optionValues, &linkState ) );

			const auto throwLinkError = [&]( const std::string& prefix ) {
				std::string message = prefix;
				if ( errorLog[0] != '\0' ) message += "\n" + std::string( errorLog.data() );
				if ( infoLog[0] != '\0' ) message += "\n" + std::string( infoLog.data() );
				checkOro( cuLinkDestroy( linkState ) );
				throw std::runtime_error( message );
			};

			if ( cuLinkAddFile( linkState, CU_JIT_INPUT_FATBINARY, const_cast<char*>( bcPath.string().c_str() ), 0, nullptr, nullptr ) !=
				 CUDA_SUCCESS )
			{
				throwLinkError( "Failed to add HIPRT precompiled fatbin: " + bcPath.string() );
			}

			if ( cuLinkAddData(
					 linkState,
					 userBinaryType,
					 const_cast<char*>( bitcodeBinary.data() ),
					 bitcodeBinary.size(),
					 const_cast<char*>( "user_bitcode" ),
					 0,
					 nullptr,
					 nullptr )
				 != CUDA_SUCCESS )
			{
				throwLinkError( "Failed to add user PTX for bitcode linking" );
			}

			if ( cuLinkAddData(
					 linkState,
					 customBinaryType,
					 const_cast<char*>( customFuncBitcodeBinary.data() ),
					 customFuncBitcodeBinary.size(),
					 const_cast<char*>( "hiprt_custom_funcs" ),
					 0,
					 nullptr,
					 nullptr )
				 != CUDA_SUCCESS )
			{
				throwLinkError( "Failed to add HIPRT custom-function PTX for bitcode linking" );
			}

			void*  linkedImage = nullptr;
			size_t linkedSize  = 0;
			if ( cuLinkComplete( linkState, &linkedImage, &linkedSize ) != CUDA_SUCCESS )
			{
				throwLinkError( "Failed to complete bitcode linking" );
			}

			binary.assign( reinterpret_cast<const char*>( linkedImage ), linkedSize );
			checkOro( cuLinkDestroy( linkState ) );

			if ( useDiskCache ) cacheBinaryToFile( binary, diskCacheName );
		}

		checkOro( cuModuleLoadData( &module, binary.data() ) );
		m_moduleCache[cacheKey] = module;
	}

	for ( const char* funcName : funcNames )
	{
		CUfunction func = nullptr;
		checkOro( cuModuleGetFunction( &func, module, funcName ) );
		functions.push_back( func );
	}
}

void Compiler::setCacheDir( const std::filesystem::path& cacheDirectory )
{
	if ( !cacheDirectory.empty() ) m_cacheDirectory = cacheDirectory;
}

std::string Compiler::kernelNameSufix( const std::string& traits )
{
	const std::string delimiter = "::";
	std::string		  result	= traits.substr( traits.find_last_of( delimiter ) + 1 );
	result						= std::regex_replace( result, std::regex( ">| " ), "" );
	result						= std::regex_replace( result, std::regex( "<|," ), "_" );
	return result;
}

std::string
Compiler::readSourceCode( const std::filesystem::path& path, std::optional<std::vector<std::filesystem::path>> includes )
{
	std::ifstream file( path );
	if ( !file.is_open() )
	{
		const std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
		throw std::runtime_error( msg );
	}

	file.seekg( 0, std::ifstream::end );
	const size_t size = static_cast<size_t>( file.tellg() );
	file.seekg( 0, std::ifstream::beg );

	std::string src;
	if ( includes )
	{
		std::string line;
		while ( std::getline( file, line ) )
		{
			if ( line.find( "#include" ) != std::string::npos )
			{
				const size_t pa = line.find( "<" );
				const size_t pb = line.find( ">" );
				includes.value().push_back( line.substr( pa + 1, pb - pa - 1 ) );
			}
			src += line + '\n';
		}
	}
	else
	{
		src.resize( size, ' ' );
		file.read( &src[0], size );
	}

	return src;
}

void Compiler::addCommonOpts( Context& context, std::vector<const char*>& opts, bool extended )
{
	if ( !extended ) opts.push_back( "--use_fast_math" );

	const uint32_t rtip = context.getRtip();
	if ( rtip > 0 )
	{
		m_rtipStr = "-DHIPRT_RTIP=" + std::to_string( rtip );
		opts.push_back( m_rtipStr.c_str() );
	}

#if defined( HIPRT_CUDA_INCLUDE_DIR )
	opts.push_back( "-I" HIPRT_CUDA_INCLUDE_DIR );
#endif
	opts.push_back( "-std=c++17" );
}

std::string Compiler::buildFunctionTableBitcode(
	Context& context, uint32_t numGeomTypes, uint32_t numRayTypes, const std::vector<hiprtFuncNameSet>& funcNameSets )
{
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	std::vector<const char*> options;
	addCommonOpts( context, options, true );

	std::string includePath = "-I" + Utility::getRootDir().string();
	options.push_back( includePath.c_str() );

	std::string archOpt = getNvrtcArchOpt( context );
	options.push_back( archOpt.c_str() );
	options.push_back( "--device-c" );

	std::string bitcodeDef = "-DHIPRT_BITCODE_LINKING";
	options.push_back( bitcodeDef.c_str() );

	std::string src = "#include <hiprt/hiprt_device.h>\n";
	addCustomFuncsSwitchCase( src, funcNameSets, numGeomTypes, numRayTypes );

	std::vector<const char*> funcNames;
	nvrtcProgram			 prog = nullptr;
	buildProgram( funcNames, src, "hiprt_bitcode_custom_funcs.cu", headers, includeNames, options, prog );
	std::string binary = getNvrtcCompiledBinary( prog );
	checkOrortc( nvrtcDestroyProgram( &prog ) );
	if ( binary.empty() )
	{
		binary = compileSourceToCubin( context, src, { "-DHIPRT_BITCODE_LINKING" }, "hiprt_bitcode_custom_funcs" );
	}
	return binary;
}

std::filesystem::path Compiler::getBitcodePath()
{
	const std::string filename = "hiprt" + std::string( HIPRT_VERSION_STR ) + "_nv_lib.fatbin";
	return findArtifactPath(
		{ Utility::getCurrentDir() / filename,
		  Utility::getRootDir() / "dist/bin/Release" / filename,
		  Utility::getRootDir() / "dist/bin/Debug" / filename,
		  Utility::getRootDir() / "hiprt/bitcodes" / filename } );
}

std::filesystem::path Compiler::findArtifactPath( const std::vector<std::filesystem::path>& candidates )
{
	for ( const auto& candidate : candidates )
	{
		if ( std::filesystem::exists( candidate ) ) return candidate;
	}

	std::string message = "Unable to locate precompiled HIPRT artifact. Checked:";
	for ( const auto& candidate : candidates )
		message += "\n  " + candidate.string();
	throw std::runtime_error( message );
}

bool Compiler::isCachedFileUpToDate( const std::filesystem::path& cachedFile, const std::filesystem::path& moduleName )
{
	if ( !std::filesystem::exists( cachedFile ) ) return false;
	if ( !std::filesystem::exists( moduleName ) ) return true;
	return std::filesystem::last_write_time( moduleName ) < std::filesystem::last_write_time( cachedFile );
}

void Compiler::addCustomFuncsSwitchCase(
	std::string&								 extSrc,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::string intersectFuncDef =
		"HIPRT_DEVICE bool intersectFunc( uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, "
		"const hiprtRay& ray, void* payload, hiprtHit& hit )\n{\n\tconst uint32_t index = tableHeader.numGeomTypes * rayType + "
		"geomType;\n\t[[maybe_unused]] const void* data = tableHeader.funcDataSets[index].intersectFuncData;\n\tswitch ( index "
		") \n\t{\n";
	std::string filterFuncDef =
		"HIPRT_DEVICE bool filterFunc( uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, const "
		"hiprtRay& ray, void* payload, const hiprtHit& hit )\n{\n\tconst uint32_t index = tableHeader.numGeomTypes * rayType + "
		"geomType;\n\t[[maybe_unused]] const void* data = tableHeader.funcDataSets[index].filterFuncData;\n\tswitch ( index ) "
		"\n\t{\n";
	std::string funcDecls;
	if ( funcNameSets )
	{
		for ( uint32_t i = 0; i < numRayTypes; ++i )
		{
			for ( uint32_t j = 0; j < numGeomTypes; ++j )
			{
				const uint32_t k = numGeomTypes * i + j;
				if ( funcNameSets.value()[k].intersectFuncName != nullptr )
				{
					const std::string intersectFuncName = funcNameSets.value()[k].intersectFuncName;
					if ( !intersectFuncName.empty() )
					{
						funcDecls += "__device__ bool " + intersectFuncName +
									 "( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit );\n";
						intersectFuncDef += "\t\tcase " + std::to_string( k ) + ": { return " + intersectFuncName +
											"( ray, data, payload, hit ); }\n";
					}
				}
				if ( funcNameSets.value()[k].filterFuncName != nullptr )
				{
					const std::string filterFuncName = funcNameSets.value()[k].filterFuncName;
					if ( !filterFuncName.empty() )
					{
						funcDecls += "__device__ bool " + filterFuncName +
									 "( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit );\n";
						filterFuncDef += "\t\tcase " + std::to_string( k ) + ": { return " + filterFuncName +
										 "( ray, data, payload, hit ); }\n";
					}
				}
			}
		}
	}

	intersectFuncDef += "\t\t default: { return false; }\n\t}\n}\n";
	filterFuncDef += "\t\t default: { return false; }\n\t}\n}\n";
	extSrc += "\n" + funcDecls + "\n" + intersectFuncDef + "\n" + filterFuncDef;
}

std::string Compiler::getCacheFilename(
	Context&									 context,
	const std::string&							 src,
	const std::filesystem::path&				 moduleName,
	std::optional<std::vector<const char*>>		 options,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::string driverVersion = context.getDriverVersion();
	std::string deviceName	  = context.getDeviceName();
	deviceName				  = deviceName.substr( 0, deviceName.find( ":" ) );

	std::string moduleHash = moduleName.string() + src;
	moduleHash			   = Utility::format( "%08x", Utility::hashString( moduleHash ) );

	std::string optionHash = moduleName.string();
	if ( funcNameSets )
	{
		for ( uint32_t i = 0; i < numRayTypes; ++i )
		{
			for ( uint32_t j = 0; j < numGeomTypes; ++j )
			{
				const uint32_t k = numGeomTypes * i + j;
				if ( funcNameSets.value()[k].intersectFuncName != nullptr )
					optionHash += funcNameSets.value()[k].intersectFuncName;
				if ( funcNameSets.value()[k].filterFuncName != nullptr ) optionHash += funcNameSets.value()[k].filterFuncName;
			}
		}
	}

	if ( options )
	{
		optionHash.append( "\n" );
		for ( const auto& option : options.value() )
			optionHash += option + std::string( "\n" );
	}
	optionHash = Utility::format( "%08x", Utility::hashString( optionHash ) );

	return moduleHash + "-" + optionHash + ".v." + deviceName + "." + driverVersion + "_" +
		   std::to_string( 8 * sizeof( void* ) ) + ".bin";
}

std::string Compiler::loadCacheFileToBinary( const std::string& cacheName )
{
	long long checksumValue = 0;
	{
		const std::filesystem::path path = m_cacheDirectory / ( cacheName + ".check" );
		std::ifstream				  file( path, std::ios::in | std::ios::binary );
		if ( !file.is_open() )
		{
			const std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		file.read( reinterpret_cast<char*>( &checksumValue ), sizeof( long long ) );
	}

	if ( checksumValue == 0 ) throw std::runtime_error( "Checksum is zero" );

	std::string binary;
	{
		const std::filesystem::path path = m_cacheDirectory / cacheName;
		std::ifstream				  file( path, std::ios::in | std::ios::binary | std::ios::ate );
		if ( !file.is_open() )
		{
			const std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		const size_t binarySize = static_cast<size_t>( file.tellg() );
		file.clear();
		file.seekg( 0, std::ios::beg );
		binary.resize( binarySize );
		file.read( binary.data(), binary.size() );
	}

	const long long hash = Utility::hashString( binary );
	if ( hash != checksumValue )
	{
		const std::string msg = Utility::format( "Checksum doesn't match %llx : %llx", hash, checksumValue );
		throw std::runtime_error( msg );
	}

	return binary;
}

void Compiler::cacheBinaryToFile( const std::string& binary, const std::string& cacheName )
{
	{
		const std::filesystem::path path = m_cacheDirectory / cacheName;
		std::ofstream				  file( path, std::ios::out | std::ios::binary );
		if ( !file.is_open() )
		{
			const std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		file.write( binary.data(), binary.size() );
	}

	const long long hash = Utility::hashString( binary );
	{
		const std::filesystem::path path = m_cacheDirectory / ( cacheName + ".check" );
		std::ofstream				  file( path, std::ios::out | std::ios::binary );
		if ( !file.is_open() )
		{
			const std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		file.write( reinterpret_cast<const char*>( &hash ), sizeof( long long ) );
	}
}
} // namespace hiprt
