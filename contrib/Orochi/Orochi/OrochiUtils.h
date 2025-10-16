//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once
#include <Orochi/Orochi.h>
#include <mutex>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <optional>

#if defined( GNUC )
#include <signal.h>
#endif

template<typename T, typename U>
constexpr void OROASSERT( T&& exp, [[maybe_unused]] U&& placeholder ) noexcept
{
	if( static_cast<bool>( std::forward<T>( exp ) ) != true )
	{

#if defined( _WIN32 )
		__debugbreak();
#elif defined( GNUC )
		raise( SIGTRAP );
#else
		;
#endif
	}
}

class OrochiUtils
{
  public:
	struct int4
	{
		int x, y, z, w;
	};

	OrochiUtils() = default;
	OrochiUtils(const OrochiUtils&) = delete;
    OrochiUtils& operator=(const OrochiUtils&) = delete;
    OrochiUtils(OrochiUtils&&) = delete;
    OrochiUtils& operator=(OrochiUtils&&) = delete;
	~OrochiUtils();

	// unload all the modules internally created during functions like getFunctionFromPrecompiledBinary/getFunction
	// good practice to call it just before cudaCtxDestroy, just to avoid any potential memory leak.
	void unloadKernelCache();

	cudaFunction_t getFunctionFromPrecompiledBinary( const std::string& path, const std::string& funcName );

	// this function is like 'getFunctionFromPrecompiledBinary' but instead of giving a path to a file, we give the data directly.
	// ( use the script convert_binary_to_array.py to convert the .hipfb to a C-array. )
	cudaFunction_t getFunctionFromPrecompiledBinary_asData( const unsigned char* data, size_t dataSizeInBytes, const std::string& funcName );

	cudaFunction_t getFunctionFromFile( int device, const char* path, const char* funcName, std::vector<const char*>* opts );
	cudaFunction_t getFunctionFromString( int device, const char* source, const char* path, const char* funcName, std::vector<const char*>* opts, int numHeaders, const char** headers, const char** includeNames );
	cudaFunction_t getFunction( int device, const char* code, const char* path, const char* funcName, std::vector<const char*>* opts, int numHeaders = 0, const char** headers = 0, const char** includeNames = 0, CUmodule* loadedModule = 0 );

	static bool readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes = 0 );
	static void getData( int device, const char* code, const char* path, std::vector<const char*>* opts, std::vector<char>& dst );
	static int getProgram( int device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, nvrtcProgram* prog );
	static void getModule( int device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, CUmodule* moduleOut );
	static void launch1D( cudaFunction_t func, int nx, const void** args, int wgSize = 64, unsigned int sharedMemBytes = 0, cudaStream_t stream = 0 );
	static void launch2D( cudaFunction_t func, int nx, int ny, const void** args, int wgSizeX = 8, int wgSizeY = 8, unsigned int sharedMemBytes = 0, cudaStream_t stream = 0 );


	struct CompressedBuffer {
		const unsigned char* data = nullptr; // compressed data
		size_t size = 0; // size in byte of 'data'
		size_t uncompressedSize = 0; // size of byte of the uncompressed data.
	};
	struct RawBuffer {
		const unsigned char* data = nullptr;
		size_t size = 0;
	};
	static void HandlePrecompiled(std::vector<unsigned char>& out, const CompressedBuffer& buffer);
	static void HandlePrecompiled(std::vector<unsigned char>& out, const RawBuffer& buffer);
	static void HandlePrecompiled(std::vector<unsigned char>& out, const unsigned char* rawData, size_t rawData_sizeByte, std::optional<size_t> uncompressed_sizeByte=std::nullopt);

	template<typename T>
	static void malloc( T*& ptr, size_t n )
	{
		cudaError e = cudaMalloc( (void **)&ptr, sizeof( T ) * n );
		OROASSERT( e == cudaSuccess, 0 );
	}

	template<typename T>
	static void free( T* ptr )
	{
		cudaFree( (void *)ptr );
	}

	static void memset( void* ptr, int val, size_t n )
	{
		cudaError e = cudaMemset( ptr, val, n );
		OROASSERT( e == cudaSuccess, 0 );
	}

	static void memsetAsync( void* ptr, int val, size_t n, cudaStream_t stream )
	{
		CUresult e = cuMemsetD8Async( reinterpret_cast<CUdeviceptr>(ptr), val, n, stream );
		OROASSERT( e == cudaSuccess, 0 );
	}

	template<typename T>
	static void copyHtoD( T* dst, const T* src, size_t n )
	{
		cudaError e = cuMemcpyHtoD( (intptr_t)dst, (void*)src, sizeof( T ) * n );
		OROASSERT( e == cudaSuccess, 0 );
	}

	template<typename T>
	static void copyDtoH( T* dst, T* src, size_t n )
	{
		cudaError e = cuMemcpyDtoH( (void*)dst, (intptr_t)src, sizeof( T ) * n );
		OROASSERT( e == cudaSuccess, 0 );
	}

	template<typename T>
	static void copyDtoD( T* dst, T* src, size_t n )
	{
		cudaError e = cuMemcpyDtoD( (intptr_t)dst, (intptr_t)src, sizeof( T ) * n );
		OROASSERT( e == cudaSuccess, 0 );
	}

	template<typename T>
	static void copyHtoDAsync( T* dst, T* src, size_t n, cudaStream_t stream )
	{
		cudaError e = cuMemcpyHtoDAsync( (intptr_t)dst, (void*)src, sizeof( T ) * n, stream );
		OROASSERT( e == cudaSuccess, 0 );
	}

	template<typename T>
	static void copyDtoHAsync( T* dst, T* src, size_t n, cudaStream_t stream )
	{
		cudaError e = cuMemcpyDtoHAsync( (void*)dst, (intptr_t)src, sizeof( T ) * n, stream );
		OROASSERT( e == cudaSuccess, 0 );
	}

	template<typename T>
	static void copyDtoDAsync( T* dst, T* src, size_t n, cudaStream_t stream )
	{
		CUresult e = cuMemcpyDtoDAsync( (intptr_t)dst, (intptr_t)src, sizeof( T ) * n, stream );
		OROASSERT( e == cudaSuccess, 0 );
	}

	static void waitForCompletion( cudaStream_t stream = 0 )
	{
		auto e = cudaStreamSynchronize( stream );
		OROASSERT( e == cudaSuccess, 0 );
	}

  public:
	std::string m_cacheDirectory = "./cache/";
	std::recursive_mutex m_mutex;

	struct FunctionModule {
		cudaFunction_t function;
		CUmodule module;
	};

	std::unordered_map<std::string, FunctionModule> m_kernelMap;
};

class OroStopwatch
{
  public:
	OroStopwatch( cudaStream_t stream )
	{
		m_stream = stream;
		cudaEventCreateWithFlags( &m_start, cudaEventDefault );
		cudaEventCreateWithFlags( &m_stop, cudaEventDefault );
	}
	~OroStopwatch()
	{
		cudaEventDestroy( m_start );
		cudaEventDestroy( m_stop );
	}

	void start() { cudaEventRecord( m_start, m_stream ); }
	void stop() { cudaEventRecord( m_stop, m_stream ); }

	float getMs()
	{
		cudaEventSynchronize( m_stop );
		float ms = 0;
		cudaEventElapsedTime( &ms, m_start, m_stop );
		return ms;
	}

  public:
	cudaStream_t m_stream;
	cudaEvent_t m_start;
	cudaEvent_t m_stop;
};