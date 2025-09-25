cmake_minimum_required(VERSION 3.0)

# 设置项目名称
project(bellhop2D Fortran)
set(CMAKE_Fortran_FLAGS "-g -Wall")

# 设置Fortran编译器
enable_language(Fortran)

# 设置源文件目录
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# 收集所有源文件
file(GLOB srcs "${SOURCE_DIR}/*.f90")

# 添加可执行目标
add_executable(bellhop2D ${srcs})

# 设置可执行文件输出目录
set_target_properties(bellhop2D PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)