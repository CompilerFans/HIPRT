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

#include <hiprt/impl/Error.h>
#include <hiprt/impl/RadixSort.h>

#include <cub/device/device_radix_sort.cuh>

namespace hiprt
{
namespace
{
class ScopedDevice final
{
  public:
	explicit ScopedDevice( int device )
	{
		checkOro( cudaGetDevice( &m_prevDevice ) );
		if ( m_prevDevice != device )
		{
			checkOro( cudaSetDevice( device ) );
			m_restore = true;
		}
	}

	~ScopedDevice()
	{
		if ( m_restore ) checkOro( cudaSetDevice( m_prevDevice ) );
	}

  private:
	int	 m_prevDevice = 0;
	bool m_restore	  = false;
};
} // namespace

RadixSort::RadixSort( int device ) : m_device( device ) {}

RadixSort::~RadixSort()
{
	if ( m_pairTempStorage == nullptr ) return;

	int	 previousDevice = 0;
	bool restoreDevice  = cudaGetDevice( &previousDevice ) == cudaSuccess && previousDevice != m_device;
	if ( restoreDevice ) cudaSetDevice( m_device );
	cudaFree( m_pairTempStorage );
	if ( restoreDevice ) cudaSetDevice( previousDevice );
}

void RadixSort::reservePairTempStorage(
	uint32_t* inputKeys,
	uint32_t* inputValues,
	uint32_t* outputKeys,
	uint32_t* outputValues,
	size_t	  size,
	cudaStream_t stream )
{
	size_t requiredBytes = 0;
	checkOro( cub::DeviceRadixSort::SortPairs(
		nullptr, requiredBytes, inputKeys, outputKeys, inputValues, outputValues, static_cast<int>( size ), 0, 32, stream ) );

	if ( requiredBytes <= m_pairTempStorageSize ) return;

	if ( m_pairTempStorage != nullptr ) checkOro( cudaFree( m_pairTempStorage ) );
	checkOro( cudaMalloc( &m_pairTempStorage, requiredBytes ) );
	m_pairTempStorageSize = requiredBytes;
}

void RadixSort::sort(
	uint32_t* inputKeys,
	uint32_t* inputValues,
	uint32_t* outputKeys,
	uint32_t* outputValues,
	size_t	  size,
	cudaStream_t stream ) noexcept
{
	ScopedDevice scopedDevice( m_device );

	reservePairTempStorage( inputKeys, inputValues, outputKeys, outputValues, size, stream );

	checkOro( cub::DeviceRadixSort::SortPairs(
		m_pairTempStorage,
		m_pairTempStorageSize,
		inputKeys,
		outputKeys,
		inputValues,
		outputValues,
		static_cast<int>( size ),
		0,
		32,
		stream ) );
}

} // namespace hiprt
