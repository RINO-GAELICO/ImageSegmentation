#include "msImageProcessor.h"

#include <iostream>
#include <cuda_runtime.h>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void msImageProcessorKernel(double *d_sdata, int *d_buckets, double *d_weightMap, float *d_msRawData,
                                       int *d_slist, double hiLTr, int nBuck1, int nBuck2,
                                       int nBuck3, int width, int height, int *d_bucNeigh,
                                       double sMins, double sigmaS, double sigmaR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    // printf("i = %d\n", i);
    int L = width * height;
    int N = 1;
    int lN = N + 2;

    if (i < L)
    {
        double yk[3];
        double Mh[3];

        int idxs = i * lN;

        for (int j = 0; j < lN; j++)
        {
            yk[j] = d_sdata[idxs + j];
        }

        // Initialize mean shift vector with 0,0,0
        for (int j = 0; j < lN; j++)
        {
            Mh[j] = 0;
        }

        double wsuml = 0;
        int cBuck1 = static_cast<int>(yk[0]) + 1;
        int cBuck2 = static_cast<int>(yk[1]) + 1;
        int cBuck3 = static_cast<int>(yk[2] - sMins) + 1;

        int cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);

        int neighborCount = 27;

        for (int j = 0; j < neighborCount; j++)
        {
            int idxd = d_buckets[cBuck + d_bucNeigh[j]];

            // printf("idxd = %d\n", idxd);
            while (idxd >= 0)
            {
                idxs = lN * idxd;

                double el = d_sdata[idxs] - yk[0];
                double diff = el * el;
                el = d_sdata[idxs + 1] - yk[1];
                diff += el * el;

                if (diff < 1.0)
                {
                    el = d_sdata[idxs + 2] - yk[2];
                    if (yk[2] > hiLTr)
                        diff = 4 * el * el;
                    else
                        diff = el * el;

                    if (diff < 1.0)
                    {
                        double weight = 1 - d_weightMap[idxd];
                        for (int k = 0; k < lN; k++)
                            Mh[k] += weight * d_sdata[idxs + k];
                        wsuml += weight;
                    }
                }
                idxd = d_slist[idxd];
            }
        }

        if (wsuml > 0)
        {
            for (int j = 0; j < lN; j++)
            {
                Mh[j] = Mh[j] / wsuml - yk[j];
            }
        }
        else
        {
            for (int j = 0; j < lN; j++)
            {
                Mh[j] = 0;
            }
        }

        double mvAbs = ((Mh[0] * Mh[0]) + (Mh[1] * Mh[1])) * sigmaS * sigmaS + (Mh[2] * Mh[2]) * sigmaR * sigmaR;

        int iterationCount = 1;
        while ((mvAbs >= EPSILON) && (iterationCount < LIMIT))
        {
            for (int j = 0; j < lN; j++)
                yk[j] += Mh[j];

            for (int j = 0; j < lN; j++)
                Mh[j] = 0;
            wsuml = 0;

            cBuck1 = static_cast<int>(yk[0]) + 1;
            cBuck2 = static_cast<int>(yk[1]) + 1;
            cBuck3 = static_cast<int>(yk[2] - sMins) + 1;
            cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);

            for (int j = 0; j < neighborCount; j++)
            {
                int idxd = d_buckets[cBuck + j];

                while (idxd >= 0)
                {
                    idxs = lN * idxd;

                    double el = d_sdata[idxs] - yk[0];
                    double diff = el * el;
                    el = d_sdata[idxs + 1] - yk[1];
                    diff += el * el;

                    if (diff < 1.0)
                    {
                        el = d_sdata[idxs + 2] - yk[2];
                        if (yk[2] > hiLTr)
                            diff = 4 * el * el;
                        else
                            diff = el * el;

                        if (diff < 1.0)
                        {
                            double weight = 1 - d_weightMap[idxd];
                            for (int k = 0; k < lN; k++)
                                Mh[k] += weight * d_sdata[idxs + k];
                            wsuml += weight;
                        }
                    }
                    idxd = d_slist[idxd];
                }
            }

            if (wsuml > 0)
            {
                for (int j = 0; j < lN; j++)
                    Mh[j] = Mh[j] / wsuml - yk[j];
            }
            else
            {
                for (int j = 0; j < lN; j++)
                    Mh[j] = 0;
            }

            mvAbs = ((Mh[0] * Mh[0]) + (Mh[1] * Mh[1])) * sigmaS * sigmaS + (Mh[2] * Mh[2]) * sigmaR * sigmaR;

            iterationCount++;
        }

        for (int j = 0; j < lN; j++)
        {
            yk[j] += Mh[j];
        }

        d_msRawData[i] = static_cast<float>(yk[2] * sigmaR);
        return;
    }
}

void msImageProcessor::Filter_cuda(float sigmaS, float sigmaR)
{
    // Host code

    // Allocate memory on the device
    // define lN
    int lN = N + 2;

    // let's use some temporary data
    double *sdata;
    sdata = new double[lN * L];
    // index the data in the 3d buckets (x, y, L)
    int *buckets;
    int *slist;
    slist = new int[L];

    // copy the scaled data
    int idxs, idxd;
    idxs = idxd = 0;

    std::cout << "This is just a test" << std::endl;
    // WE FOCUS ON GRAYSCALE
    if (N == 1)
    {
        for (int i = 0; i < L; i++)
        {
            sdata[idxs++] = (i % width) / sigmaS;
            sdata[idxs++] = (i / width) / sigmaS;
            sdata[idxs++] = data[idxd++] / sigmaR;
        }
    }

    int bucNeigh[27]; // 27 because it is 3x3x3

    double sMins;    // just for L
    double sMaxs[3]; // for all

    // we store the max values of each dimension
    //  the range of the scaled values for the intensity
    sMaxs[0] = width / sigmaS;
    sMaxs[1] = height / sigmaS;
    sMins = sMaxs[2] = sdata[2];
    idxs = 2;
    double cval;
    // find the min and max values of the intensity
    for (int i = 0; i < L; i++)
    {
        cval = sdata[idxs];
        if (cval < sMins)
            sMins = cval;
        else if (cval > sMaxs[2])
            sMaxs[2] = cval;

        idxs += lN;
    }

    int nBuck1, nBuck2, nBuck3;
    int cBuck1, cBuck2, cBuck3, cBuck;
    nBuck1 = (int)(sMaxs[0] + 3);
    nBuck2 = (int)(sMaxs[1] + 3);
    nBuck3 = (int)(sMaxs[2] - sMins + 3);
    buckets = new int[nBuck1 * nBuck2 * nBuck3];
    for (int i = 0; i < (nBuck1 * nBuck2 * nBuck3); i++)
    {
        buckets[i] = -1;
    }

    idxs = 0;
    for (int i = 0; i < L; i++)
    {
        // find bucket for current data and add it to the list
        cBuck1 = (int)sdata[idxs] + 1;
        cBuck2 = (int)sdata[idxs + 1] + 1;
        cBuck3 = (int)(sdata[idxs + 2] - sMins) + 1;
        cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);

        slist[i] = buckets[cBuck];
        buckets[cBuck] = i;

        idxs += lN;
    }

    idxd = 0;
    for (cBuck1 = -1; cBuck1 <= 1; cBuck1++)
    {
        for (cBuck2 = -1; cBuck2 <= 1; cBuck2++)
        {
            for (cBuck3 = -1; cBuck3 <= 1; cBuck3++)
            {
                bucNeigh[idxd++] = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);
            }
        }
    }
    
    double hiLTr = 80.0 / sigmaR;

    // Allocate memory on the device
    double *d_sdata;
    int *d_buckets;
    double *d_weightMap;
    int *d_slist;
    int *d_bucNeigh;
    float *d_msRawData;

    gpuErrchk(cudaMalloc(&d_sdata, sizeof(double) * lN * L));
    gpuErrchk(cudaMalloc(&d_buckets, sizeof(int) * nBuck1 * nBuck2 * nBuck3));
    gpuErrchk(cudaMalloc(&d_weightMap, sizeof(double) * L));
    gpuErrchk(cudaMalloc(&d_slist, sizeof(int) * L));
    gpuErrchk(cudaMalloc(&d_bucNeigh, sizeof(int) * 27));
    gpuErrchk(cudaMalloc(&d_msRawData, sizeof(float) * L));

    // Transfer data from host to device
    cudaMemcpy(d_sdata, sdata, sizeof(double) * lN * L, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buckets, buckets, sizeof(int) * nBuck1 * nBuck2 * nBuck3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightMap, weightMap, sizeof(double) * L, cudaMemcpyHostToDevice);
    cudaMemcpy(d_slist, slist, sizeof(int) * L, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bucNeigh, bucNeigh, sizeof(int) * 27, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel

    const int threadsPerBlock = 1024;
    const int numberBlocks = (L + threadsPerBlock - 1) / threadsPerBlock;

    // let's measure the time

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    msImageProcessorKernel<<<numberBlocks, threadsPerBlock>>>(d_sdata, d_buckets, d_weightMap, d_msRawData, d_slist,
                                                    hiLTr, nBuck1, nBuck2, nBuck3, width, height, d_bucNeigh,
                                                    sMins, sigmaS, sigmaR);

    // Check for CUDA errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
        // Handle the error appropriately
    }

    // Synchronize to ensure the kernel has completed
    cudaDeviceSynchronize();

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total GPU processing time without copying:\t" << elapsed.count() << " s" << std::endl;

    // Copy the result back to the host
    cudaMemcpy(msRawData, d_msRawData, sizeof(float) * L, cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(d_sdata);
    cudaFree(d_buckets);
    cudaFree(d_weightMap);
    cudaFree(d_slist);
    cudaFree(d_bucNeigh);
    cudaFree(d_msRawData);
    return;
}
