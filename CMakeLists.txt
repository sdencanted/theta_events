cmake_minimum_required(VERSION 3.2)

project(theta_events
VERSION   0.1
LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -DTORCH_USE_CUDA_DSA")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)


# OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  message(STATUS "OPENMP NOT FOUND")
endif()

# OpenCV
find_package (OpenCV 4.0.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# Prophesee metavision
find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)
message(STATUS "Find metavision include at ${MetavisionSDK_INCLUDE_DIRS}")
message(STATUS "Find metavision libraries: ${MetavisionSDK_LIBRARIES}")

#Torch
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "$ENV{HOME}/.local/lib/python3.8/site-packages/torch")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "$ENV{HOME}/libtorch")
find_package(Torch REQUIRED)
message(STATUS "Find Torch libs: ${TORCH_LIBRARIES}")

#jetson-utils
find_package(jetson-utils)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# compile mag library
set(theta_events_mag_SOURCE_FILES 
${PROJECT_SOURCE_DIR}/src/rm3100_spi_userspace.c
)
add_library(theta_events_mag ${theta_events_mag_SOURCE_FILES})
target_include_directories(theta_events_mag PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  theta_events_mag
  m
  gpiod
)

find_package(Boost 1.71.0)
include_directories(${Boost_INCLUDE_DIRS})

# compile evk4 library
# set(theta_events_evk4_SOURCE_FILES 
# )
# add_library(theta_events_evk4 ${theta_events_evk4_SOURCE_FILES})
# target_include_directories(theta_events_evk4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(
#   theta_events_evk4
#   metavision_hal
#   metavision_hal_discovery
#   metavision_psee_hw_layer
#   Boost::program_options
# )


# Executables

add_executable(rm3100_test_cpp src/rm3100_test.cpp)
target_link_libraries(rm3100_test_cpp theta_events_mag)

add_executable(drdy_on src/drdy_on.cpp)
target_link_libraries(drdy_on theta_events_mag)

add_executable(drdy_off src/drdy_off.cpp)
target_link_libraries(drdy_off theta_events_mag)
# add_executable(poll_drdy src/poll_drdy.cpp)
# target_link_libraries(poll_drdy theta_events_mag)

add_executable(evk4_test src/evk4_test.cpp)
target_link_libraries(
  evk4_test
  MetavisionSDK::core 
  MetavisionSDK::driver 
  MetavisionSDK::ui

)

add_executable(evk4_kronecker src/evk4_kronecker.cpp src/key.cu)
target_link_libraries(
  evk4_kronecker
  ${TORCH_LIBRARIES}
  MetavisionSDK::core 
  MetavisionSDK::driver 
  MetavisionSDK::ui
  ${OpenMP_LIBS}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  jetson-utils
)
# add_executable(evk4_accumulator src/evk4_accumulator.cpp)
# target_link_libraries(
#   evk4_accumulator
#   MetavisionSDK::core 
#   MetavisionSDK::driver 
#   MetavisionSDK::ui
#   theta_events_mag
# )