#ifndef SPMV_INTERNAL_PAGERANK_COMMON_H
#define SPMV_INTERNAL_PAGERANK_COMMON_H

#include "spmv/pagerank.h"

#include <vector>

namespace spmv {

std::vector<int> pagerank_find_dangling_nodes(const CSRMatrix* adj_matrix);
void pagerank_normalize(float* ranks, int n);

}  // namespace spmv

#endif  // SPMV_INTERNAL_PAGERANK_COMMON_H
