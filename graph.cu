
#include <assert.h>
#include <stdio.h>

#include "graph.h"

struct COOGraph {
    unsigned int numVertices;
    unsigned int numEdges;
    unsigned int* src;
    unsigned int* dst;
};

COOGraph* readCOOGraph(const char* fileName) {

    COOGraph* cooGraph = (COOGraph*) malloc(sizeof(COOGraph));

    // Initialize fields
    FILE* fp = fopen(fileName, "r");
    int x = 1;
    x |= fscanf(fp, "%u", &cooGraph->numVertices);
    x |= fscanf(fp, "%u", &cooGraph->numEdges);
    cooGraph->src = (unsigned int*) malloc(cooGraph->numEdges*sizeof(unsigned int));
    cooGraph->dst = (unsigned int*) malloc(cooGraph->numEdges*sizeof(unsigned int));

    // Read the nonzeros
    for(unsigned int i = 0; i < cooGraph->numEdges; ++i) {
        x |= fscanf(fp, "%u", &cooGraph->src[i]);
        x |= fscanf(fp, "%u", &cooGraph->dst[i]);
    }

    return cooGraph;

}

void freeCOOGraph(COOGraph* cooGraph) {
    free(cooGraph->src);
    free(cooGraph->dst);
    free(cooGraph);
}

CSRGraph* coo2csr(COOGraph* cooGraph) {

    CSRGraph* csrGraph = (CSRGraph*) malloc(sizeof(CSRGraph));;

    // Initialize fields
    csrGraph->numVertices = cooGraph->numVertices;
    csrGraph->numEdges = cooGraph->numEdges;
    csrGraph->srcPtrs = (unsigned int*) malloc((csrGraph->numVertices + 1)*sizeof(unsigned int));
    csrGraph->dst = (unsigned int*) malloc(csrGraph->numEdges*sizeof(unsigned int));

    // Histogram vertices
    memset(csrGraph->srcPtrs, 0, (csrGraph->numVertices + 1)*sizeof(unsigned int));
    for(unsigned int i = 0; i < cooGraph->numEdges; ++i) {
        unsigned int vertex = cooGraph->src[i];
        csrGraph->srcPtrs[vertex]++;
    }

    // Prefix sum srcPtrs
    unsigned int sumBeforeNextRow = 0;
    for(unsigned int vertex = 0; vertex < csrGraph->numVertices; ++vertex) {
        unsigned int sumBeforeRow = sumBeforeNextRow;
        sumBeforeNextRow += csrGraph->srcPtrs[vertex];
        csrGraph->srcPtrs[vertex] = sumBeforeRow;
    }
    csrGraph->srcPtrs[csrGraph->numVertices] = sumBeforeNextRow;

    // Bin the nonzeros
    for(unsigned int i = 0; i < cooGraph->numEdges; ++i) {
        unsigned int vertex = cooGraph->src[i];
        unsigned int edgeIdx = csrGraph->srcPtrs[vertex]++;
        csrGraph->dst[edgeIdx] = cooGraph->dst[i];
    }

    // Restore srcPtrs
    for(unsigned int vertex = csrGraph->numVertices - 1; vertex > 0; --vertex) {
        csrGraph->srcPtrs[vertex] = csrGraph->srcPtrs[vertex - 1];
    }
    csrGraph->srcPtrs[0] = 0;

    return csrGraph;

}

void freeCSRGraph(CSRGraph* csrGraph) {
    free(csrGraph->srcPtrs);
    free(csrGraph->dst);
    free(csrGraph);
}

CSRGraph* createCSRGraphFromFile(const char* fileName) {
    COOGraph* cooGraph = readCOOGraph(fileName);
    CSRGraph* csrGraph = coo2csr(cooGraph);
    freeCOOGraph(cooGraph);
    return csrGraph;
}

CSRGraph* createEmptyCSRGraphOnGPU(unsigned int numVertices, unsigned int numEdges) {

    CSRGraph csrGraphShadow;
    csrGraphShadow.numVertices = numVertices;
    csrGraphShadow.numEdges = numEdges;
    cudaMalloc((void**) &csrGraphShadow.srcPtrs, (numVertices + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &csrGraphShadow.dst, numEdges*sizeof(unsigned int));

    CSRGraph* csrGraph;
    cudaMalloc((void**) &csrGraph, sizeof(CSRGraph));
    cudaMemcpy(csrGraph, &csrGraphShadow, sizeof(CSRGraph), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    return csrGraph;

}

void freeCSRGraphOnGPU(CSRGraph* csrGraph) {
    CSRGraph csrGraphShadow;
    cudaMemcpy(&csrGraphShadow, csrGraph, sizeof(CSRGraph), cudaMemcpyDeviceToHost);
    cudaFree(csrGraphShadow.srcPtrs);
    cudaFree(csrGraphShadow.dst);
    cudaFree(csrGraph);
}

void copyCSRGraphToGPU(CSRGraph* csrGraph_h, CSRGraph* csrGraph_d) {
    CSRGraph csrGraphShadow;
    cudaMemcpy(&csrGraphShadow, csrGraph_d, sizeof(CSRGraph), cudaMemcpyDeviceToHost);
    assert(csrGraphShadow.numVertices == csrGraph_h->numVertices);
    assert(csrGraphShadow.numEdges == csrGraph_h->numEdges);
    cudaMemcpy(csrGraphShadow.srcPtrs, csrGraph_h->srcPtrs, (csrGraph_h->numVertices + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrGraphShadow.dst, csrGraph_h->dst, csrGraph_h->numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

COOMatrix* createEmptyCOOMatrix(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix *cooMatrix = (COOMatrix *) malloc(sizeof(COOMatrix));
    cooMatrix->numRows = numRows;
    cooMatrix->numCols = numCols;
    cooMatrix->nnz = 0;
    cooMatrix->capacity = capacity;
    cooMatrix->rowIdxs = (unsigned int *) calloc(1, capacity * sizeof(unsigned int));
    cooMatrix->colIdxs = (unsigned int *) malloc( capacity * sizeof(unsigned int));
    cooMatrix->values = (float *) malloc( capacity * sizeof(float));
    return cooMatrix;
}

void freeCOOMatrix(COOMatrix* cooMatrix) {
    free(cooMatrix->rowIdxs);
    free(cooMatrix->colIdxs);
    free(cooMatrix->values);
    free(cooMatrix);
}


void insertionSort(unsigned int *key1, unsigned int *key2, float *data, unsigned int N) {
    for(unsigned int i = 1; i < N; ++i) {
        unsigned int j = i;
        while(j > 0 && (key1[j - 1] > key1[j] || key1[j - 1] == key1[j] && key2[j - 1] > key2[j])) {
            unsigned int tmpKey1 = key1[j]; key1[j] = key1[j - 1]; key1[j - 1] = tmpKey1;
            unsigned int tmpKey2 = key2[j]; key2[j] = key2[j - 1]; key2[j - 1] = tmpKey2;
            float tmpData = data[j]; data[j] = data[j - 1]; data[j - 1] = tmpData;
            --j;
        }
    }
}

void insertionSortCOOMatrix(COOMatrix* cooMatrix) {
    insertionSort(cooMatrix->rowIdxs, cooMatrix->colIdxs, cooMatrix->values, cooMatrix->nnz);
}

void quickSort(unsigned int *key1, unsigned int *key2, float *data, unsigned int start, unsigned int end) {
    if((end - start + 1) > 1) {
        unsigned int left = start, right = end;
        unsigned int pivot1 = key1[right];
        unsigned int pivot2 = key2[right];
        while(left <= right) {
            while(key1[left] < pivot1 || key1[left] == pivot1 && key2[left] < pivot2) {
                left = left + 1;
            }
            while(key1[right] > pivot1 || key1[right] == pivot1 && key2[right] > pivot2) {
                right = right - 1;
            }
            if(left <= right) {
                unsigned int tmpKey1 = key1[left]; key1[left] = key1[right]; key1[right] = tmpKey1;
                unsigned int tmpKey2 = key2[left]; key2[left] = key2[right]; key2[right] = tmpKey2;
                float tmpData = data[left]; data[left] = data[right]; data[right] = tmpData;
                left = left + 1;
                right = right - 1;
            }
        }
        quickSort(key1, key2, data, start, right);
        quickSort(key1, key2, data, left, end);
    }
}

void quickSortCOOMatrix(COOMatrix* cooMatrix) {
    quickSort(cooMatrix->rowIdxs, cooMatrix->colIdxs, cooMatrix->values, 0, cooMatrix->nnz - 1);
}

COOMatrix* createEmptyCOOMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int capacity) {

    COOMatrix cooMatrixShadow;
    cooMatrixShadow.numRows = numRows;
    cooMatrixShadow.numCols = numCols;
    cooMatrixShadow.nnz = 0;
    cooMatrixShadow.capacity = capacity;
    cudaMalloc((void**) &cooMatrixShadow.rowIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooMatrixShadow.colIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooMatrixShadow.values, capacity*sizeof(float));

    COOMatrix* cooMatrix;
    cudaMalloc((void**) &cooMatrix, sizeof(COOMatrix));
    cudaMemcpy(cooMatrix, &cooMatrixShadow, sizeof(COOMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    return cooMatrix;

}

void clearCOOMatrixOnGPU(COOMatrix* cooMatrix) {
    COOMatrix cooMatrixShadow;
    cudaMemcpy(&cooMatrixShadow, cooMatrix, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    cudaMemset(cooMatrixShadow.rowIdxs, 0, cooMatrixShadow.nnz*sizeof(unsigned int));
    cudaMemset(cooMatrixShadow.colIdxs, 0, cooMatrixShadow.nnz*sizeof(unsigned int));
    cudaMemset(cooMatrixShadow.values, 0, cooMatrixShadow.nnz*sizeof(unsigned int));
    cudaMemset(&cooMatrix->nnz, 0, sizeof(unsigned int));
}

void freeCOOMatrixOnGPU(COOMatrix* cooMatrix) {
    COOMatrix cooMatrixShadow;
    cudaMemcpy(&cooMatrixShadow, cooMatrix, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    cudaFree(cooMatrixShadow.rowIdxs);
    cudaFree(cooMatrixShadow.colIdxs);
    cudaFree(cooMatrixShadow.values);
    cudaFree(cooMatrix);
}

void copyCOOMatrixFromGPU(COOMatrix* cooMatrix_d, COOMatrix* cooMatrix_h) {
    COOMatrix cooMatrixShadow;
    cudaMemcpy(&cooMatrixShadow, cooMatrix_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    assert(cooMatrix_h->numRows == cooMatrixShadow.numRows);
    assert(cooMatrix_h->numCols == cooMatrixShadow.numCols);
    assert(cooMatrix_h->capacity >= cooMatrixShadow.nnz);
    cooMatrix_h->nnz = cooMatrixShadow.nnz;
    cudaMemcpy(cooMatrix_h->rowIdxs, cooMatrixShadow.rowIdxs, cooMatrixShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooMatrix_h->colIdxs, cooMatrixShadow.colIdxs, cooMatrixShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooMatrix_h->values, cooMatrixShadow.values, cooMatrixShadow.nnz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

