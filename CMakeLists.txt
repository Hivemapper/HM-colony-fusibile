cmake_minimum_required (VERSION 3.9)
project (fusibile)

# Enable C++17 globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Support IDEs: https://cliutils.gitlab.io/modern-cmake/chapters/features/ides.html
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake-default-targets")

find_package(OpenCV REQUIRED )
find_package(CUDA 6.0 REQUIRED ) # For Cuda Managed Memory and c++17

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

# from https://en.wikipedia.org/wiki/CUDA#GPUs_supported
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v --compiler-options -Wall -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++17 --compiler-options -Wall -gencode arch=compute_52,code=sm_52)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=compute_52) # For debugging only

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++17)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
    #add_definitions(-march=native)
endif()


# for YouCompleteMe
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    fusibile
    src/BinaryCvMat.cpp
    src/cameraGeometryUtils.h
    src/vector_operations.h
    src/camera.h
    src/globalstate.h
    src/algorithmparameters.h
    src/cameraparameters.h
    src/linestate.h
    src/displayUtils.h
    src/mathUtils.h
    src/fileIoUtils.h
    src/fusibile.cu
    src/main.cpp
    )
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
find_package(OpenMP)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(fusibile
	${OpenCV_LIBS}
	OpenMP::OpenMP_CXX
	)


include(FeatureSummary)
feature_summary(WHAT ALL)
