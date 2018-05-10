cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
#find_package( OpenCV REQUIRED )
find_package(OpenCV REQUIRED PATHS /usr/local/OpenCV NO_DEFAULT_PATH) # manual specification of the OpenCVConfig.cmake path is necessary since otherwise, cmake assigns /opt/local a higher priority and ocv2.4 will be used
add_executable( DisplayImage 1D_Array.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
