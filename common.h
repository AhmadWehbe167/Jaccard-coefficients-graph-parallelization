
#ifndef _COMMON_H_
#define _COMMON_H_

#include "graph.h"

void jaccard_gpu0(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d);
void jaccard_gpu1(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d);
void jaccard_gpu2(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d);
void jaccard_gpu3(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d);

#endif

