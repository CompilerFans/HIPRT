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

#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace hiprt
{

class RadixSort final
{
  public:
	explicit RadixSort( int device );
	~RadixSort();

	RadixSort( const RadixSort& ) = delete;
	RadixSort& operator=( const RadixSort& ) = delete;

	void sort(
		uint32_t* inputKeys,
		uint32_t* inputValues,
		uint32_t* outputKeys,
		uint32_t* outputValues,
		size_t	  size,
		cudaStream_t stream ) noexcept;

  private:
	void reservePairTempStorage(
		uint32_t* inputKeys,
		uint32_t* inputValues,
		uint32_t* outputKeys,
		uint32_t* outputValues,
		size_t	  size,
		cudaStream_t stream );

	int	   m_device			 = 0;
	void*  m_pairTempStorage	 = nullptr;
	size_t m_pairTempStorageSize = 0;
};
} // namespace hiprt
