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

#include <cub/device/device_radix_sort.cuh>

#include <hiprt/impl/Error.h>
#include <hiprt/impl/RadixSort.h>

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

size_t getRadixSortPairsTemporaryStorageSize( size_t count )
{
	size_t requiredBytes = 0;
	checkOro( cub::DeviceRadixSort::SortPairs(
		nullptr,
		requiredBytes,
		static_cast<const uint32_t*>( nullptr ),
		static_cast<uint32_t*>( nullptr ),
		static_cast<const uint32_t*>( nullptr ),
		static_cast<uint32_t*>( nullptr ),
		static_cast<int>( count ),
		0,
		32,
		0 ) );

	return requiredBytes;
}

void radixSortPairs(
	int				device,
	void*			temporaryStorage,
	size_t			temporaryStorageSize,
	uint32_t*		inputKeys,
	uint32_t*		inputValues,
	uint32_t*		outputKeys,
	uint32_t*		outputValues,
	size_t			count,
	cudaStream_t	stream ) noexcept
{
	ScopedDevice scopedDevice( device );

	checkOro( cub::DeviceRadixSort::SortPairs(
		temporaryStorage,
		temporaryStorageSize,
		inputKeys,
		outputKeys,
		inputValues,
		outputValues,
		static_cast<int>( count ),
		0,
		32,
		stream ) );
}

} // namespace hiprt
