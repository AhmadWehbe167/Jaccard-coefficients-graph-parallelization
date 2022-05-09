
#ifndef __GRAPH_H_
#define __GRAPH_H_

struct CSRGraph {
    unsigned int numVertices;
    unsigned int numEdges;
    unsigned int* srcPtrs;
    unsigned int* dst;
};

CSRGraph* createCSRGraphFromFile(const char* fileName);
void freeCSRGraph(CSRGraph* csrGraph);

CSRGraph* createEmptyCSRGraphOnGPU(unsigned int numVertices, unsigned int numEdges);
void freeCSRGraphOnGPU(CSRGraph* csrGraph);

void copyCSRGraphToGPU(CSRGraph* csrGraph_h, CSRGraph* csrGraph_d);

struct COOMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int nnz;
    unsigned int capacity;
    unsigned int* rowIdxs;
    unsigned int* colIdxs;
    float* values;
};

COOMatrix* createEmptyCOOMatrix(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void freeCOOMatrix(COOMatrix* cooMatrix);

void insertionSortCOOMatrix(COOMatrix* cooMatrix);
void quickSortCOOMatrix(COOMatrix* cooMatrix);

COOMatrix* createEmptyCOOMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void freeCOOMatrixOnGPU(COOMatrix* cooMatrix);

void clearCOOMatrixOnGPU(COOMatrix* cooMatrix);
void copyCOOMatrixFromGPU(COOMatrix* cooMatrix_d, COOMatrix* cooMatrix_h);

#endif

