cmake_minimum_required(VERSION 3.3)
project(template)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    bridge/Makefile
    externals/argvparser/include/argvparser.h
    externals/argvparser/src/argvparser.cpp
    externals/EasyBMP/include/EasyBMP.h
    externals/EasyBMP/include/EasyBMP_BMP.h
    externals/EasyBMP/include/EasyBMP_DataStructures.h
    externals/EasyBMP/include/EasyBMP_VariousBMPutilities.h
    externals/EasyBMP/src/EasyBMP.cpp
    externals/liblinear-1.94/blas/blas.h
    externals/liblinear-1.94/blas/blasp.h
    externals/liblinear-1.94/blas/daxpy.c
    externals/liblinear-1.94/blas/ddot.c
    externals/liblinear-1.94/blas/dnrm2.c
    externals/liblinear-1.94/blas/dscal.c
    externals/liblinear-1.94/linear.cpp
    externals/liblinear-1.94/linear.h
    externals/liblinear-1.94/predict.c
    externals/liblinear-1.94/train.c
    externals/liblinear-1.94/tron.cpp
    externals/liblinear-1.94/tron.h
    include/classifier.h
    include/matrix.h
    include/matrix.hpp
    src/task2.cpp
    compare.py
    deps.mk
    Makefile)

add_executable(template ${SOURCE_FILES})