
# Image Segmentation and Border Highlighting

This C++ program performs image segmentation using mean-shift algorithm and highlights the borders of the segmented regions.


# OpenMeanShift SERIAL

This repository contains an implementation of the Mean Shift algorithm in SERIAL. The code is designed to run on a CPU, and the following steps guide you through the process of building and running the program.

## Prerequisites

- C++ compiler

## Compile the program on Perlmutter:

module load cpu
salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu --account=m3930
mkdir build
cd build
cmake ..
make

## Run

After successfully building the project, you can run the executable. Inside build run the following command:

```bash
main/main "<input image> <area> <sigmaS> <sigmaR> <implementation>
```

Example:

```bash
main/main ../data/512_cat.png 200 16 16 "SERIAL"
```


# OpenMeanShift OPEN_MP

## Prerequisites

- C++ compiler with OpenMP support

## Compile the program on Perlmutter:

Same as Serial but add this command for # of Threads:

```bash
export OMP_NUM_THREADS=4
```

Change the number before running the executable.

## Run

After successfully building the project, you can run the executable.

```bash
main/main "<input image> <area> <sigmaS> <sigmaR> "OPEN_MP"
```


# OpenMeanShift CUDA

## Prerequisites

Before you start, make sure you have access to an environment with NVIDIA GPU.

salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --account=m3930

Set up the environment like this:

```bash
    module load PrgEnv-nvidia
    export CC=cc
    export CXX=CC 
 ```

Then, once your environment is set up, then:
```bash
    mkdir build  
    cd build  
    cmake ../  
    make
 ```


## Run

After successfully building the project, you can run the executable. Depending on the program arguments, the application switches between different implementations of the Mean Shift algorithm.

```bash
main/main "<input image> <area> <sigmaS> <sigmaR> <implementation>
```
