#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <stdint.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <omp.h>

#include "def.h"

namespace clustering {

extern int g_k;                     // global param: #clusters
extern f32 g_mae;                   // global param: mean absolute error
extern f32 g_mse;                   // global param: mean square   error
extern f64 g_tot_wc_time;           // global param: total wall clock time (s)

extern int g_iter;                  // global param: #iterations
extern f64 g_init_wc_time;          // global param: init wall clock time (s)
extern f64 g_iter_wc_time;          // global param: iter wall clock time (s)
extern f64 g_kpp_wc_time;           // global param: k-freqitems++ wall clock time (s)

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path does not exist
    char *path);                        // input path

// -----------------------------------------------------------------------------
template<class DType>
DType* read_sparse_data(            // read sparse data (binary) from disk
    int   n,                            // number of data points
    const char *addr_data,              // address of data set
    u64   *datapos)                     // data position (return)
{
    double start_time = omp_get_wtime();
    std::ios::sync_with_stdio(false);

    FILE *fp = fopen(addr_data, "rb");
    if (!fp) { printf("ERROR: cannot open %s\n", addr_data); exit(1); }

    // read the start position of each data
    fread(datapos, sizeof(u64), n+1, fp);
    
    // read dataset
    u64 N = datapos[n];
    DType *dataset = new DType[N];
    fread(dataset, sizeof(DType), N, fp);
    fclose(fp);
    
    double loading_time = omp_get_wtime() - start_time;
    printf("\nn=%d, N=%lu, time=%.2lf seconds, path=%s\n\n", n, N, 
        loading_time, addr_data);
    return dataset;
}

// -----------------------------------------------------------------------------
template<class DType>
float jaccard_dist(                 // calc jaccard distance
    int   n_data,                       // number of data dimensions
    int   n_mode,                       // number of mode dimensions
    const DType *data,                  // data point
    const int   *mode)                  // mode
{
    int overlap = 0, i = 0, j = 0; // i for data, j for mode
    while (i < n_data && j < n_mode) {
        if (data[i] < mode[j]) ++i;
        else if (mode[j] < data[i]) ++j;
        else { ++overlap; ++i; ++j; }
    }
    return 1.0f - (float) overlap / (n_data + n_mode - overlap);
}

// -----------------------------------------------------------------------------
template<class DType>
float jaccard_dist2(                // calc jaccard distance
    int   n_data,                       // number of data dimensions
    int   n_seed,                       // number of seed dimensions
    const DType *data,                  // data point
    const DType *seed)                  // seed point
{
    int overlap = 0, i = 0, j = 0; // i for data, j for seed
    while (i < n_data && j < n_seed) {
        if (data[i] < seed[j]) ++i;
        else if (seed[j] < data[i]) ++j;
        else { ++overlap; ++i; ++j; }
    }
    return 1.0f - (float) overlap / (n_data + n_seed - overlap);
}

// -----------------------------------------------------------------------------
inline int get_length(// get the length of pos
    int   id,                           // input id
    const u64 *pos)                     // pos array
{
    return int(pos[id+1] - pos[id]);
}

// -----------------------------------------------------------------------------
template<class DType>
int distinct_coord_and_freq(        // get max freq, distinct coords & freqs
    u64   total_num,                    // total number of coordinates
    DType *arr,                         // store all coordinates (allow modify)
    DType *coord,                       // distinct coordinates (return)
    int   *freq,                        // frequency (return)
    int   &cnt)                         // counter for #distinct (return)
{
    // sort all coordinates in ascending order
    std::sort(arr, arr + total_num);
    
    // get the distinct coordinates and their frequencies (sequential)
    int max_freq = 0, last = 0, this_freq = -1;
    cnt = 0;
    for (size_t i = 1; i < total_num; ++i) {
        if (arr[i] != arr[i-1]) {
            this_freq = i - last;
            coord[cnt] = arr[i-1]; freq[cnt] = this_freq;
            if (this_freq > max_freq) max_freq = this_freq;
            
            last = i; ++cnt;
        }
    }
    // deal with the last element of arr
    this_freq = total_num - last;
    coord[cnt] = arr[total_num-1]; freq[cnt] = this_freq;
    if (this_freq > max_freq) max_freq = this_freq;
    ++cnt;
    
    return max_freq;
}

// -----------------------------------------------------------------------------
float uniform(                      // gen a random variable from uniform distr.
    float start,                        // start position
    float end);                         // end position

} // end namespace clustering
