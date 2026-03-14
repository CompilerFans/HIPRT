cmake_minimum_required(VERSION 3.19)
project(hiprt LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89;86;80;75")

option(NO_UNITTEST "Don't build unit tests" OFF)

list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDAToolkit_PRIMARY_INCLUDE_DIR)
string(REPLACE "\\" "\\\\" HIPRT_CUDA_INCLUDE_DIR_ESCAPED "${CUDAToolkit_PRIMARY_INCLUDE_DIR}")

add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
add_definitions(-DHIPRT_PUBLIC_REPO)

if(WIN32)
	add_definitions(-D__WINDOWS__)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4305 /wd4018 /wd4996 /Zc:__cplusplus")
elseif(UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

function(copy_dir src_dir dst_dir pattern)
	file(GLOB_RECURSE files "${src_dir}/${pattern}")
	foreach(file ${files})
		file(RELATIVE_PATH rel_file ${src_dir} ${file})
		set(target "${dst_dir}/${rel_file}")
		get_filename_component(target_dir ${target} DIRECTORY)
		file(MAKE_DIRECTORY ${target_dir})
		file(COPY ${file} DESTINATION ${target_dir})
	endforeach()
endfunction()

function(get_version file)
	file(STRINGS ${file} lines)
	list(GET lines 0 major)
	list(GET lines 1 minor)
	list(GET lines 2 patch)
	list(GET lines 3 hash)
	set(major ${major} PARENT_SCOPE)
	set(minor ${minor} PARENT_SCOPE)
	set(patch ${patch} PARENT_SCOPE)
	set(hash "0x${hash}" PARENT_SCOPE)
endfunction()

function(write_version_info in_file header_file version_file version_str_out)
	if(NOT EXISTS ${version_file})
		message(FATAL_ERROR "Version.txt file missing!")
	endif()
	if(NOT EXISTS ${in_file})
		message(FATAL_ERROR "${in_file} file is missing!")
	endif()

	get_version(${version_file})

	file(READ ${in_file} header_content)

	math(EXPR HIPRT_VERSION "${major} * 1000 + ${minor}")
	string(LENGTH "${HIPRT_VERSION}" HIPRT_VERSION_LEN)
	if(${HIPRT_VERSION_LEN} LESS 5)
		math(EXPR HIPRT_VERSION_PAD "5 - ${HIPRT_VERSION_LEN}")
		string(REPEAT "0" ${HIPRT_VERSION_PAD} HIPRT_VERSION_PADDED)
		set(version_str "${HIPRT_VERSION_PADDED}${HIPRT_VERSION}")
	else()
		set(version_str "${HIPRT_VERSION}")
	endif()

	string(REPLACE "@HIPRT_MAJOR_VERSION@" "${major}" header_content "${header_content}")
	string(REPLACE "@HIPRT_MINOR_VERSION@" "${minor}" header_content "${header_content}")
	string(REPLACE "@HIPRT_PATCH_VERSION@" "${patch}" header_content "${header_content}")
	string(REPLACE "@HIPRT_HASH_VERSION@" "${hash}" header_content "${header_content}")
	string(REPLACE "@HIPRT_VERSION_STR@" "\"${version_str}\"" header_content "${header_content}")
	string(REPLACE "@HIPRT_API_VERSION@" "${HIPRT_VERSION}" header_content "${header_content}")

	file(WRITE ${header_file} "${header_content}")
	set(${version_str_out} ${version_str} PARENT_SCOPE)
endfunction()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;DebugGpu" CACHE STRING "Configs" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUGGPU "")
set(CMAKE_EXE_LINKER_FLAGS_DEBUGGPU "")

set(BASE_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin)
file(MAKE_DIRECTORY "${BASE_OUTPUT_DIR}/Debug")
file(MAKE_DIRECTORY "${BASE_OUTPUT_DIR}/Release")

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} Debug Release RelWithDebInfo MinSizeRel)
	string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
endforeach()

if(NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
endif()

set(version_str_ "UNDEF")
write_version_info("${CMAKE_CURRENT_SOURCE_DIR}/hiprt/hiprt.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/hiprt.h" "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" version_str_)

set(HIPRT_NAME "hiprt${version_str_}")

add_library(${HIPRT_NAME} SHARED)

target_compile_definitions(${HIPRT_NAME} PRIVATE
	HIPRT_EXPORTS
	HIPRT_CUDA_INCLUDE_DIR="${HIPRT_CUDA_INCLUDE_DIR_ESCAPED}"
)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set_target_properties(${HIPRT_NAME} PROPERTIES OUTPUT_NAME "${HIPRT_NAME}64D")
else()
	set_target_properties(${HIPRT_NAME} PROPERTIES OUTPUT_NAME "${HIPRT_NAME}64")
endif()

if(WIN32)
	target_link_libraries(${HIPRT_NAME} PRIVATE version)
endif()

target_include_directories(${HIPRT_NAME} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDAToolkit_INCLUDE_DIRS}
)

file(GLOB_RECURSE hiprt_sources
	"${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.inl"
)
target_sources(${HIPRT_NAME} PRIVATE ${hiprt_sources})

target_link_libraries(${HIPRT_NAME} PRIVATE
	CUDA::cudart
	CUDA::cuda_driver
	CUDA::nvrtc
)

install(TARGETS ${HIPRT_NAME} DESTINATION bin)

file(GLOB HIPRT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.h")
install(FILES ${HIPRT_HEADERS} DESTINATION include/hiprt)

file(GLOB HIPRT_IMPL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/impl/*.h")
list(REMOVE_ITEM HIPRT_IMPL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/impl/bvh_build_array.h")
install(FILES ${HIPRT_IMPL_HEADERS} DESTINATION include/hiprt/impl)

if(NOT NO_UNITTEST)
	add_executable(unittest)

	target_link_libraries(unittest PRIVATE
		${HIPRT_NAME}
		CUDA::cudart
		CUDA::cuda_driver
		CUDA::nvrtc
	)

	if(WIN32)
		target_compile_options(unittest PRIVATE /wd4244)
		target_link_libraries(unittest PRIVATE version)
	endif()

	if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
		set_target_properties(unittest PROPERTIES OUTPUT_NAME "unittest64D")
	else()
		set_target_properties(unittest PROPERTIES OUTPUT_NAME "unittest64")
	endif()

	target_include_directories(unittest PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
		${CUDAToolkit_INCLUDE_DIRS}
		${CMAKE_CURRENT_SOURCE_DIR}/contrib/gtest-1.6.0
		${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/include
	)
	target_compile_definitions(unittest PRIVATE HIPRT_CUDA_INCLUDE_DIR="${HIPRT_CUDA_INCLUDE_DIR_ESCAPED}")

	if(UNIX)
		target_link_libraries(unittest PRIVATE pthread dl embree4 tbb)
		target_link_directories(unittest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/linux)
	endif()

	file(GLOB_RECURSE unittest_sources
		"${CMAKE_CURRENT_SOURCE_DIR}/test/hiprtT*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/test/hiprtT*.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/test/shared.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/test/CornellBox.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/contrib/gtest-1.6.0/gtest-all.cc"
	)
	target_sources(unittest PRIVATE ${unittest_sources})

	if(WIN32)
		target_link_directories(unittest PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/win
			${CMAKE_CURRENT_SOURCE_DIR}/contrib/bin/win64
		)
		copy_dir(${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/win ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin/Release "*.dll")
		copy_dir(${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/win ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin/Debug "*.dll")
		target_link_libraries(unittest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/win/embree4.lib)
	endif()
endif()
