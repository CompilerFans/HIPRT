PWD_PATH=`pwd`
export LD_LIBRARY_PATH="$PWD_PATH/../contrib/embree/linux:${LD_LIBRARY_PATH}"
export HIPRT_DISABLE_RUNTIME_KERNEL_CACHE="${HIPRT_DISABLE_RUNTIME_KERNEL_CACHE:-0}"
../dist/bin/Release/unittest64 --width=512 --height=512 --referencePath=../test/references/ --gtest_filter=-*PerformanceTest* --gtest_output=xml:../result.xml
