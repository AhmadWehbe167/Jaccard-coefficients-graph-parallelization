#include "common.h"
#include "timer.h"
#include <limits.h>

#define BLOCK_DIM 1024
#define FACTOR 32
#define BATCH 32
#define ITER 10
#define SMEM_SIZE ITER*BLOCK_DIM
#define ANSI_COLOR_MAGENTA    "\x1b[35m"

__global__ void jaccard_kernel2(CSRGraph* csrGraph,  COOMatrix* cooMatrix, unsigned int* numCommonNeighbors, unsigned int* neighborsOfNeighbors){
    __shared__ unsigned int numCommonNeighbors_s[SMEM_SIZE];
    __shared__ unsigned int numNeighborsOfNeighbors;
    
    unsigned int startIndex = blockIdx.x * csrGraph->numVertices;

    if(threadIdx.x==0) {
        numNeighborsOfNeighbors=0;
    }
    for(int i=0; i < ITER; ++i){
        numCommonNeighbors_s[i * BLOCK_DIM + threadIdx.x] = 0;
    }
    __syncthreads();
    for(unsigned int v = 0; v < FACTOR; ++v) {
        unsigned int vertex = FACTOR * blockIdx.x + v;
        if(vertex < csrGraph->numVertices) {
            for(unsigned int e = csrGraph->srcPtrs[vertex]; e < csrGraph->srcPtrs[vertex + 1]; e += blockDim.x / BATCH){
                unsigned int edge = e + threadIdx.x / BATCH;
                if(edge < csrGraph->srcPtrs[vertex + 1]) {
                    int neighbor = csrGraph->dst[edge];
                    for(long neighborEdge = (long) csrGraph->srcPtrs[neighbor + 1]; neighborEdge > (long) csrGraph->srcPtrs[neighbor]; neighborEdge -= BATCH ) {
                        if( neighborEdge - ( threadIdx.x % BATCH ) - 1 > csrGraph->srcPtrs[neighbor]) {
                            unsigned int neighborOfNeighbor = csrGraph->dst[neighborEdge - ( threadIdx.x % BATCH ) - 1];
                            if(neighborOfNeighbor > vertex) {
                                unsigned int oldVal;
                                if(neighborOfNeighbor < SMEM_SIZE) {
                                    oldVal = atomicAdd(&(numCommonNeighbors_s[neighborOfNeighbor]), 1);
                                } else {
                                    oldVal = atomicAdd(&(numCommonNeighbors[startIndex + neighborOfNeighbor]), 1);
                                }
                                if( oldVal == 0 ) {
                                    neighborsOfNeighbors[startIndex + atomicAdd(&(numNeighborsOfNeighbors), 1)] = neighborOfNeighbor;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
            __syncthreads();
            for(unsigned int i = 0; i < numNeighborsOfNeighbors; i += blockDim.x) {
                if(i + threadIdx.x < numNeighborsOfNeighbors){
                    unsigned int vertex2;
                    vertex2 = neighborsOfNeighbors[startIndex + i + threadIdx.x]; 
                    if(vertex2 < SMEM_SIZE){
                        if(numCommonNeighbors_s[vertex2] > 0) {
                            unsigned int numNeighbors = csrGraph->srcPtrs[vertex + 1] - csrGraph->srcPtrs[vertex];
                            unsigned int numNeighbors2 = csrGraph->srcPtrs[vertex2 + 1] - csrGraph->srcPtrs[vertex2];
                            float jaccardSimilarity = ((float) numCommonNeighbors_s[vertex2])/(numNeighbors + numNeighbors2 - numCommonNeighbors_s[vertex2]);
                            unsigned int j = atomicAdd(&(cooMatrix->nnz), 1);
                            cooMatrix->rowIdxs[j] = vertex;
                            cooMatrix->colIdxs[j] = vertex2;
                            cooMatrix->values[j] = jaccardSimilarity;
                            numCommonNeighbors_s[vertex2] = 0;
                        }
                    }else{
                        if(numCommonNeighbors[startIndex + vertex2] > 0) {
                            unsigned int numNeighbors = csrGraph->srcPtrs[vertex + 1] - csrGraph->srcPtrs[vertex];
                            unsigned int numNeighbors2 = csrGraph->srcPtrs[vertex2 + 1] - csrGraph->srcPtrs[vertex2];
                            float jaccardSimilarity = ((float) numCommonNeighbors[startIndex + vertex2])/(numNeighbors + numNeighbors2 - numCommonNeighbors[startIndex + vertex2]);
                            unsigned int j = atomicAdd(&(cooMatrix->nnz), 1);
                            cooMatrix->rowIdxs[j] = vertex;
                            cooMatrix->colIdxs[j] = vertex2;
                            cooMatrix->values[j] = jaccardSimilarity;
                            numCommonNeighbors[startIndex + vertex2] = 0;
                        }
                    }         
                }
            }
        }
        __syncthreads();
        if(threadIdx.x==0) numNeighborsOfNeighbors=0;
    }
}

void jaccard_gpu2(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d) {
    Timer timer;

    // Configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numBlocks = ( csrGraph->numVertices + FACTOR - 1 ) / FACTOR;

    //allocate mem
    startTime(&timer);
    unsigned int* numCommonNeighbors;
    unsigned int* neighborsOfNeighbors;
    cudaMalloc((void**) &numCommonNeighbors, numBlocks*csrGraph->numVertices*sizeof(unsigned int)+1);
    cudaMalloc((void**) &neighborsOfNeighbors, numBlocks*csrGraph->numVertices*sizeof(unsigned int)+1);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Arrays allocation time");

    //Call Kernel.cu
    startTime(&timer);
    jaccard_kernel2 <<< numBlocks, numThreadsPerBlock >>> (csrGraph_d, cooMatrix_d, numCommonNeighbors, neighborsOfNeighbors);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);
}