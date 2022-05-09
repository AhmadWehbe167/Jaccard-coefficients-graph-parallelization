
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "graph.h"
#include "timer.h"

void jaccard_cpu(CSRGraph* csrGraph, COOMatrix* cooMatrix) {
    unsigned int* neighborsOfNeighbors = (unsigned int*) malloc(csrGraph->numVertices*sizeof(unsigned int));
    unsigned int* numCommonNeighbors = (unsigned int*) malloc(csrGraph->numVertices*sizeof(unsigned int));
    unsigned int numNeighborsOfNeighbors = 0;
    memset(numCommonNeighbors, 0, csrGraph->numVertices*sizeof(unsigned int));
    for(unsigned int vertex = 0; vertex < csrGraph->numVertices; ++vertex) {
        for(unsigned int edge = csrGraph->srcPtrs[vertex]; edge < csrGraph->srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph->dst[edge];
            for(unsigned int neighborEdge = csrGraph->srcPtrs[neighbor + 1]; neighborEdge > csrGraph->srcPtrs[neighbor]; --neighborEdge) {
                unsigned int neighborOfNeighbor = csrGraph->dst[neighborEdge - 1];
                if(neighborOfNeighbor > vertex) {
                    unsigned int oldVal = numCommonNeighbors[neighborOfNeighbor]++;
                    if(oldVal == 0) {
                        neighborsOfNeighbors[numNeighborsOfNeighbors++] = neighborOfNeighbor;
                    }
                } else {
                    break;
                }
            }
        }
        for(unsigned int i = 0; i < numNeighborsOfNeighbors; ++i) {
            unsigned int vertex2 = neighborsOfNeighbors[i];
            if(numCommonNeighbors[vertex2] > 0) {
                unsigned int numNeighbors = csrGraph->srcPtrs[vertex + 1] - csrGraph->srcPtrs[vertex];
                unsigned int numNeighbors2 = csrGraph->srcPtrs[vertex2 + 1] - csrGraph->srcPtrs[vertex2];
                float jaccardSimilarity = ((float) numCommonNeighbors[vertex2])/(numNeighbors + numNeighbors2 - numCommonNeighbors[vertex2]);
                assert(cooMatrix->nnz < cooMatrix->capacity);
                unsigned int j = cooMatrix->nnz++;
                cooMatrix->rowIdxs[j] = vertex;
                cooMatrix->colIdxs[j] = vertex2;
                cooMatrix->values[j] = jaccardSimilarity;
                numCommonNeighbors[vertex2] = 0;
            }
        }
        numNeighborsOfNeighbors = 0;
    }
    free(neighborsOfNeighbors);
    free(numCommonNeighbors);
}

void verify(COOMatrix* cooMatrixGPU, COOMatrix* cooMatrixCPU, unsigned int quickVerify) {
    if(cooMatrixCPU->nnz != cooMatrixGPU->nnz) {
        printf("    \033[1;31mMismatching number of non-zeros (CPU result = %d, GPU result = %d)\033[0m\n", cooMatrixCPU->nnz, cooMatrixGPU->nnz);
        return;
    } else if(quickVerify) {
        printf("    Quick verification succeeded\n");
        printf("        This verification is not exact. For exact verification, pass the -v flag.\n");
    } else {
        printf("    Verifying result\n");
        insertionSortCOOMatrix(cooMatrixCPU);
        quickSortCOOMatrix(cooMatrixGPU);
        for(unsigned int i = 0; i < cooMatrixCPU->nnz; ++i) {
            unsigned int rowCPU = cooMatrixCPU->rowIdxs[i];
            unsigned int rowGPU = cooMatrixGPU->rowIdxs[i];
            unsigned int colCPU = cooMatrixCPU->colIdxs[i];
            unsigned int colGPU = cooMatrixGPU->colIdxs[i];
            float valCPU = cooMatrixCPU->values[i];
            float valGPU = cooMatrixGPU->values[i];
            if(rowCPU != rowGPU || colCPU != colGPU || abs(valGPU - valCPU) > 1e-6) {
                printf("        \033[1;31mMismatch detected: CPU: (%d, %d, %f), GPU: (%d, %d, %f)\033[0m\n", rowCPU, colCPU, valCPU, rowGPU, colGPU, valGPU);
                return;
            }
        }
        printf("        Verification succeeded\n");
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();
    setbuf(stdout, NULL);

    // Parse arguments
    const char* graphFile = "data/loc-brightkite_edges.txt";
    unsigned int runGPUVersion0 = 0;
    unsigned int runGPUVersion1 = 0;
    unsigned int runGPUVersion2 = 0;
    unsigned int runGPUVersion3 = 0;
    unsigned int quickVerify = 1;
    int opt;
    while((opt = getopt(argc, argv, "g:0123v")) >= 0) {
        switch(opt) {
            case 'g': graphFile = optarg;   break;
            case '0': runGPUVersion0 = 1;   break;
            case '1': runGPUVersion1 = 1;   break;
            case '2': runGPUVersion2 = 1;   break;
            case '3': runGPUVersion3 = 1;   break;
            case 'v': quickVerify = 0;      break;
            default:  fprintf(stderr, "\nUnrecognized option!\n");
                      exit(0);
        }
    }

    // Allocate memory and initialize data
    printf("Reading matrix from file: %s\n", graphFile);
    CSRGraph* csrGraph = createCSRGraphFromFile(graphFile);
    printf("Allocating COO matrices\n");
    COOMatrix* cooMatrix = createEmptyCOOMatrix(csrGraph->numVertices, csrGraph->numVertices, csrGraph->numVertices*10000);
    COOMatrix* cooMatrix_h = createEmptyCOOMatrix(csrGraph->numVertices, csrGraph->numVertices, csrGraph->numVertices*10000);

    // Compute on CPU
    printf("Running CPU version\n");
    Timer timer;
    startTime(&timer);
    jaccard_cpu(csrGraph, cooMatrix);
    stopTime(&timer);
    printElapsedTime(timer, "    CPU time", CYAN);

    if(runGPUVersion0 || runGPUVersion1 || runGPUVersion2 || runGPUVersion3) {

        // Allocate GPU memory
        startTime(&timer);
        CSRGraph* csrGraph_d = createEmptyCSRGraphOnGPU(csrGraph->numVertices, csrGraph->numEdges);
        COOMatrix* cooMatrix_d = createEmptyCOOMatrixOnGPU(cooMatrix->numRows, cooMatrix->numCols, cooMatrix->capacity);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "GPU allocation time");

        // Copy data to GPU
        startTime(&timer);
        copyCSRGraphToGPU(csrGraph, csrGraph_d);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Copy to GPU time");

        if(runGPUVersion0) {

            printf("Running GPU version 0\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 0
            startTime(&timer);
            jaccard_gpu0(csrGraph, csrGraph_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 0)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        if(runGPUVersion1) {

            printf("Running GPU version 1\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 1
            startTime(&timer);
            jaccard_gpu1(csrGraph, csrGraph_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 1)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        if(runGPUVersion2) {

            printf("Running GPU version 2\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 2
            startTime(&timer);
            jaccard_gpu2(csrGraph, csrGraph_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 2)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        if(runGPUVersion3) {

            printf("Running GPU version 3\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 3
            startTime(&timer);
            jaccard_gpu3(csrGraph, csrGraph_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 3)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        // Free GPU memory
        startTime(&timer);
        freeCSRGraphOnGPU(csrGraph_d);
        freeCOOMatrixOnGPU(cooMatrix_d);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "GPU deallocation time");

    }

    // Free memory
    freeCSRGraph(csrGraph);
    freeCOOMatrix(cooMatrix);

    return 0;

}

