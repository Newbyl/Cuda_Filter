#!/bin/bash

CUDA_FILE=$1
CUDA_PATH="/opt/cuda"
HIPIFY_PATH="/opt/rocm/hip/bin/hipify-clang"
OPENCV_PATH="/usr/include/opencv4"
HIPCC_PATH="/opt/rocm/bin/hipcc"



$HIPIFY_PATH  $CUDA_FILE --cuda-path $CUDA_PATH -I $OPENCV_PATH  -- -stdlib=libc++
$HIPCC_PATH "$CUDA_FILE.hip" -I "$OPENCV_PATH" -lopencv_imgcodecs -lopencv_core -o "$CUDA_FILE.out"


