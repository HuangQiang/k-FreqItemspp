#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>

#include "def.h"
#include "util.h"

namespace clustering {

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for input data
    int   did,                          // input data id
    int   s_len,                        // length of last seed
    const DType *seed,                  // input seed
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    float &nn_dist)                     // nn_dist (return)
{
    int   d_len = get_length(did, datapos);
    const DType *data = dataset + datapos[did];
    
    float dist = jaccard_dist2<DType>(d_len, s_len, data, seed);
    if (nn_dist > dist) nn_dist = dist;
}

// -----------------------------------------------------------------------------
template<class DType>
void update_dist_and_prob(          // update nn_dist and prob by last seed
    int   n,                            // number of data points
    int   id,                           // last seed id
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *weights,               // weights of data set
    float *nn_dist,                     // nn_dist (return)
    float *prob)                        // probability (return)
{
    // get last seed
    int   s_len = get_length(id, datapos);
    const DType *seed = dataset + datapos[id];
    
    // update nn_dist for the local data (use OpenMP by default)
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        update_nn_dist<DType>(i, s_len, seed, dataset, datapos, nn_dist[i]);
    }
    
    // update the probability array
    prob[0] = weights[0] * SQR(nn_dist[0]);
    for (int i = 1; i < n; ++i) {
        prob[i] = prob[i-1] + weights[i] * SQR(nn_dist[i]);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void get_k_seeds(                   // get k seeds based on distinct ids
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int   *distinct_ids,          // k distinct ids
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    std::vector<int>().swap(seedset);
    std::vector<u64>().swap(seedpos);
    
    // init seedpos (sequential, cannot parallel)
    int id = -1, len = -1;
    seedpos.resize(k+1); seedpos[0] = 0;
    for (int i = 0; i < k; ++i) {
        id  = distinct_ids[i];          // get data id
        len = get_length(id, datapos);  // get the length of data
        const DType *data = dataset + datapos[id]; // get data 
        
        seedpos[i+1] = seedpos[i] + len;
    }
    
    // get k seeds from dataset
    seedset.resize(seedpos[k]);
    for (int i = 0; i < k; ++i) {
        id  = distinct_ids[i];          // get data id
        len = get_length(id, datapos);  // get the length of data
        const DType *data = dataset + datapos[id]; // get data 
        
        // add this data into seedset
        int *seed = seedset.data() + seedpos[i];
        std::copy(data, data+len, seed);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void kmeanspp_seeding(              // init k centers by k-means++
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *weights,               // weights of data set
    int   *distinct_ids,                // k distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    srand(RANDOM_SEED); // fix a random seed
    
    // -------------------------------------------------------------------------
    //  init nn_dist array & probability array
    // -------------------------------------------------------------------------
    float *nn_dist = new float[n];
    for (int i = 0; i < n; ++i) nn_dist[i] = MAX_FLOAT;
        
    float *prob = new float[n];
    prob[0] = (float) weights[0];
    for (int i = 1; i < n; ++i) prob[i] = prob[i-1] + weights[i];
    
    // -------------------------------------------------------------------------
    //  sample the first center uniformly at random
    // -------------------------------------------------------------------------
    float val = uniform(0.0f, prob[n-1]);
    int   id  = std::lower_bound(prob, prob+n, val) - prob;
    distinct_ids[0] = id;
    
    // -------------------------------------------------------------------------
    //  sample the remaining (k-1) centers by D^2 sampling
    // -------------------------------------------------------------------------
    for (int i = 1; i < k; ++i) {
        // update nn_dist and prob by last_seed
        update_dist_and_prob<DType>(n, id, dataset, datapos, weights, nn_dist, prob);
        
        // sample the i-th center (id) by D^2 sampling
        val = uniform(0.0f, prob[n-1]);
        id  = std::lower_bound(prob, prob+n, val) - prob;
        distinct_ids[i] = id;
        
#ifdef DEBUG_INFO
        if ((i+1)%100 == 0) printf("k-FreqItems++ Seeding: %d/%d\n", i+1, k);
#endif
    }
    // -------------------------------------------------------------------------
    //  get the global seedset and seedpos by the k distinct ids
    // -------------------------------------------------------------------------
    get_k_seeds<DType>(n, k, distinct_ids, dataset, datapos, seedset, seedpos);

    // release space
    delete[] nn_dist;
    delete[] prob;
}

// -----------------------------------------------------------------------------
template<class DType>
int get_label(                      // get label (0,k-1) for input data
    int   k,                            // number of seeds
    int   n_data,                       // length of input data
    const DType *data,                  // input data
    const int   *seedset,               // seed set
    const u64   *seedpos)               // seed position
{
    int   label = 0;
    float nn_dist = -1.0f;
    
    for (int i = 0; i < k; ++i) {
        int n_seed = get_length(i, seedpos);
        const int *seed = seedset + seedpos[i];
        
        float dist = jaccard_dist<DType>(n_data, n_seed, data, seed);
        if (nn_dist < 0 || dist < nn_dist) { nn_dist = dist; label = i; }
    }
    return label;
}

// -----------------------------------------------------------------------------
template<class DType>
void exact_assign_data(             // exact sparse data assginment
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos,               // seed position
    int   *labels)                      // cluster labels for dataset (return)
{
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int n_data = get_length(i, datapos);
        const DType *data = dataset + datapos[i];
        
        labels[i] = get_label<DType>(k, n_data, data, seedset, seedpos);
    }
}

// -----------------------------------------------------------------------------
u64 labels_to_index(                // convert labels into index and index_pos
    int   n,                            // number of labels
    int   k,                            // number of centers
    const int *labels,                  // cluster labels for data points
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos);          // bin position (return)

// -----------------------------------------------------------------------------
int labels_to_bins(                 // labels to bins & re-number labels
    int n,                              // number of data points
    int k,                              // number of centers
    int *labels,                        // cluster labels for data (return)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos);          // bin position (return)

// -----------------------------------------------------------------------------
template<class DType>
int frequent_items(                 // find frequent items as a seed
    int   num,                          // number of point IDs in a bin
    int   max_len,                      // max length for a seed
    float alpha,                        // global \alpha \in (0,1)
    const int   *bin,                   // bin
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    DType *seed)                        // a seed (return)
{
    // deal with the special case with a single data
    if (num == 1) {
        int   id = bin[0];
        const DType *data = dataset + datapos[id]; // get data
        int   len = get_length(id, datapos);       // get data len
        
        len = std::min(max_len, len);
        std::copy(data, data+len, seed);
        return len;
    }
    
    // consider the case with multiple data
    // get the total number of coordinates in this bin
    u64 tot_num = 0UL;
    for (int i = 0; i < num; ++i) {
        int id = bin[i]; tot_num += get_length(id, datapos);
    }
    
    // init an array to store all coordinates sequentially
    DType *arr = new DType[tot_num];
    int len = 0; 
    u64 cnt = 0UL;
    for (int i = 0; i < num; ++i) {
        int   id = bin[i];
        const DType *data = dataset + datapos[id]; // get data
        len = get_length(id, datapos);         // get data len
        
        // copy the coordinates of this data to the array
        std::copy(data, data+len, arr+cnt);
        cnt += len;
    }
    assert(cnt == tot_num);
    
    // get the distinct coordinates and their frequencies
    DType *coord = new DType[tot_num];
    int   *freq  = new int[tot_num];
    int n = 0; // number of distinct coordinates
    int max_freq = distinct_coord_and_freq<DType>(tot_num, arr, coord, freq, n);

    // sequentially get the high frequent coordinates as seed 
    int threshold = (int) ceil((double) max_freq*alpha);
    len = 0; // number of coordinates for seed
    for (int i = 0; i < n; ++i) {
        if (freq[i] >= threshold) {
            seed[len++] = coord[i];
            if (len >= max_len) break;
        }
    }
    // release space
    delete[] arr; delete[] coord; delete[] freq;
    
    return len;
}

// -----------------------------------------------------------------------------
template<class DType>
void bins_to_seeds(                 // convert bins into seeds
    int   n,                            // number of data points
    int   k,                            // number of bins (and seeds)
    int   avg_d,                        // average dimension of data points
    float alpha,                        // \alpha \in (0,1)
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *binset,                // bin set
    const u64   *binpos,                // bin position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    // clear seedset and seedpos
    std::vector<int>().swap(seedset);
    std::vector<u64>().swap(seedpos);
    
    // determine k seeds
    int max_len = 100*avg_d; // TODO: the factor 100 can be tuned
    DType *seeds = new DType[(u64)k*max_len];
    seedpos.resize(k+1); seedpos[0] = 0;
#pragma omp parallel for
    for (int i = 0; i < k; ++i) {
        const int *bin = binset + binpos[i];  // get a bin
        int num = get_length(i, binpos); // get # point ID's in a bin
        
        seedpos[i+1] = frequent_items<DType>(num, max_len, alpha, bin, 
            dataset, datapos, seeds+i*max_len);
    }
    
    // determine seedpos by accumulating the size of each seed
    for (int i = 1; i <= k; ++i) seedpos[i] += seedpos[i-1];
    
    // convert seeds into seedset
    seedset.resize(seedpos[k]);
    int *seedset_ptr = seedset.data();
#pragma omp parallel for
    for (int i = 0; i < k; ++i) {
        int num = seedpos[i+1] - seedpos[i];
        DType *seed = seeds + (u64) i*max_len;
        std::copy(seed, seed+num, seedset_ptr+seedpos[i]);
    }
    delete[] seeds;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_jaccard_dist(            // calc jaccard dist between data & seed
    int   did,                          // data id
    int   sid,                          // label (seed id)
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos)               // seed position
{
    int n_data = get_length(did, datapos);
    const DType *data = dataset + datapos[did];
    
    int n_seed = get_length(sid, seedpos);
    const int *seed = seedset + seedpos[sid];
    
    return jaccard_dist<DType>(n_data, n_seed, data, seed);
}

// -----------------------------------------------------------------------------
template<class DType>
void calc_stat_by_seeds(            // calc statistics by seeds
    int   n,                            // number of data points
    int   k,                            // number of clusters
    const int   *labels,                // cluster labels for data points
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos,               // seed position
    float &mae,                         // mean absolute error (return)
    float &mse)                         // mean square   error (return)
{
    // calc the jaccard distance for local data to its nearest seed
    float *dist = new float[n];
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        dist[i] = calc_jaccard_dist<DType>(i, labels[i], dataset, datapos, 
            seedset, seedpos);
    }
    
    // sequentially calc mae and mse for clusters
    mae = 0.0f; mse = 0.0f;
    float dis = -1.0f;
    for (int i = 0; i < n; ++i) {
        dis = dist[i]; mae += dis; mse += SQR(dis);
    }
    mae /= n; mse /= n;
    
    delete[] dist;
}

} // end namespace clustering
