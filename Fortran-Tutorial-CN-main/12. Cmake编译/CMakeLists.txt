cmake_minimum_required(VERSION 3.10)
project(main Fortran)  # 项目名简化

set(CMAKE_Fortran_FLAGS "-g -Wall")

# 直接编译所有文件，不单独建库
add_executable(main
    main.f90
    FINITE_DERIVATIVE.f90
    FD_SCHEME.f90
    POISSON_SOLVER.f90
    ADVECTION_DIFFUSION.f90
    CONVECTION_DIFFUSION.f90
)