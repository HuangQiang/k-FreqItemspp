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
#include "seeding.h"

namespace clustering {

// -----------------------------------------------------------------------------
//  KFreqItems: k-freqitems clustering for sparse data over Jaccard distance
// -----------------------------------------------------------------------------
template<class DType>
class KFreqItems {
public:
    KFreqItems(                     // constructor
        int   n,                        // number of data points
        int   max_iter,                 // maximum iteration
        float alpha,                    // global alpha
        const char  *folder,            // output folder
        const DType *dataset,           // data set
        const u64   *datapos);          // data position
    
    // -------------------------------------------------------------------------
    ~KFreqItems();                      // destructor
    
    // -------------------------------------------------------------------------
    void display();                 // display parameters
    
    // -------------------------------------------------------------------------
    int clustering(                 // k-freqitems clustering
        int k);                         // #clusters (specified by users)
    
protected:
    int   n_;                       // number of data points
    int   max_iter_;                // maximum iteration
    float alpha_;                   // global \alpha
    const DType *dataset_;          // data set
    const u64   *datapos_;          // data position
    char  folder_[200];             // output folder
    
    int   avg_d_;                   // average dimension of sparse data
    int   *labels_;                 // cluster labels
    std::vector<int> binset_;       // bin set
    std::vector<u64> binpos_;       // bin position
    std::vector<int> seedset_;      // seed set
    std::vector<u64> seedpos_;      // seed position
    
    // -------------------------------------------------------------------------
    void free();                    // free space for local parameters
};

// -----------------------------------------------------------------------------
template<class DType>
KFreqItems<DType>::KFreqItems(      // constructor
    int   n,                            // number of data points
    int   max_iter,                     // maximum iteration
    float alpha,                        // global alpha
    const char  *folder,                // output folder
    const DType *dataset,               // data set
    const u64   *datapos)               // data position
    : n_(n), max_iter_(max_iter), alpha_(alpha), dataset_(dataset), datapos_(datapos)
{
    srand(RANDOM_SEED); // fix a random seed
    strncpy(folder_, folder, sizeof(folder_)); // init folder_
    labels_ = new int[n]; // init label_
    
    // calc avg_d, i.e., the average number of non-empty coordinates 
    avg_d_ = (int) ceil((double) datapos[n] / (double) n);
}

// -----------------------------------------------------------------------------
template<class DType>
KFreqItems<DType>::~KFreqItems()    // destructor
{
    free();
    delete[] labels_;
}

// -----------------------------------------------------------------------------
template<class DType>
void KFreqItems<DType>::free()      // free space for local parameters
{
    std::vector<int>().swap(binset_);
    std::vector<u64>().swap(binpos_);
    std::vector<int>().swap(seedset_);
    std::vector<u64>().swap(seedpos_);
}

// -----------------------------------------------------------------------------
template<class DType>
void KFreqItems<DType>::display()   // display parameters
{
    printf("The parameters of KFreqItems:\n");
    printf("n        = %d\n",   n_);
    printf("avg_d    = %d\n",   avg_d_);
    printf("max_iter = %d\n",   max_iter_);
    printf("alpha    = %g\n",   alpha_);
    printf("folder   = %s\n\n", folder_);
}

// -----------------------------------------------------------------------------
void output_iter_info(              // output info for each k-freqitems iteration
    int    k,                           // specified number of clusters
    int    iter,                        // which iteration
    int    max_iter,                    // maximum iteration
    int    K,                           // actual number of clusters
    float  mae,                         // mean absolute error
    float  mse,                         // mean square error
    double assign_wc_time,              // data assignment wall clock time
    double update_wc_time,              // seed update wall clock time
    double total_wc_time,               // total wall clock time so far
    const  char *folder)                // output folder
{
    // output binary format
    char fname[200]; sprintf(fname, "%s%d_iter_info.csv", folder, k);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    
    fprintf(fp, "%d,%f,%f,%.2lf,", K, mse, mae, total_wc_time);
    fprintf(fp, "%d,%d,%d,%.2lf+%.2lf=%.2lf\n", k, max_iter, iter, 
        assign_wc_time, update_wc_time-assign_wc_time, update_wc_time);
    fclose(fp);
}

// -----------------------------------------------------------------------------
void output_labels(                 // output labels
    int   n,                            // number of data points (local)
    int   k,                            // number of clusters
    const int *labels,                  // cluster labels [0,k-1]
    const char *folder)                 // output folder
{
    // output labels
    char fname[200]; 
    sprintf(fname, "%s%d_kFreqItems++.labels", folder, k);
    
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    fwrite(labels, sizeof(int), n, fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
void output_centers(                // output k centers as seeds
    int   k,                            // number of seeds
    const std::vector<int> &seedset,    // seed set (return)
    const std::vector<u64> &seedpos,    // seed position (return)
    const char *folder)                 // output folder
{
    char fname[100]; sprintf(fname, "%s%d_kFreqItems++.seeds", folder, k);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }

    fwrite(&k, sizeof(int), 1, fp);
    fwrite(seedpos.data(), sizeof(u64), k+1, fp);
    fwrite(seedset.data(), sizeof(int), seedpos[k], fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
template<class DType>
int KFreqItems<DType>::clustering(  // k-freqitems clustering
    int k)                              // #clusters (specified by users)
{
    double start_wc_time  = omp_get_wtime();
    
    // -------------------------------------------------------------------------
    //  k-means++ seeding: select k data points as seeds (use OpemMP by default)
    // -------------------------------------------------------------------------
    int *distinct_ids = new int[k];
    int *weights = new int[n_]; memset(weights, 1, sizeof(int)*n_);
    kmeanspp_seeding<DType>(n_, k, dataset_, datapos_, weights, distinct_ids, 
        seedset_, seedpos_);
    
    delete[] weights;
    delete[] distinct_ids;
    g_init_wc_time = omp_get_wtime() - start_wc_time;
    
#ifdef DEBUG_INFO
    printf("\nk-FreqItems++ Seeding: k=%d, init_time=%.2lf seconds\n\n", k, 
        g_init_wc_time);
#endif

    // -------------------------------------------------------------------------
    //  assignment-update iterations
    // -------------------------------------------------------------------------
    int K = k; // actual number of clusters (K <= k)
    f32 mae = -1.0f, mse = -1.0f;
    f64 assign_wc_time, update_wc_time;
    
    g_mse = MAX_FLOAT;
    for (int iter = 1; iter <= max_iter_; ++iter) {
        // data assignment (assign.cu)
        double local_start_wtime = omp_get_wtime();
        exact_assign_data<DType>(n_, K, dataset_, datapos_, seedset_.data(), 
            seedpos_.data(), labels_);
        assign_wc_time = omp_get_wtime() - local_start_wtime;
        
        // update freqitems & re-number the labels in [0,K-1] (bin.cu)
        K = labels_to_bins(n_, K, labels_, binset_, binpos_);
        
        // convert bins into seeds (assign.cuh)
        bins_to_seeds<DType>(n_, K, avg_d_, alpha_, dataset_, datapos_, 
            binset_.data(), binpos_.data(), seedset_, seedpos_);
        
        // evaluation based on new freqitems and new labels
        calc_stat_by_seeds<DType>(n_, K, labels_, dataset_, datapos_,
            seedset_.data(), seedpos_.data(), mae, mse);
        
        update_wc_time = omp_get_wtime() - local_start_wtime;
        g_tot_wc_time  = omp_get_wtime() - start_wc_time;
        
        if (mse < g_mse) {
            g_k = K; g_mae = mae; g_mse = mse; g_iter = iter;
            g_kpp_wc_time = g_tot_wc_time;
        }
        
#ifdef DEBUG_INFO
        printf("iter=%d/%d, k=%d, mse=%f, mae=%f, time=%.2lf+%.2lf=%.2lf, "
            "total_time=%.2lf\n\n", iter, max_iter_, K, mse, mae, assign_wc_time, 
            update_wc_time-assign_wc_time, update_wc_time, g_tot_wc_time);
        
        output_iter_info(k, iter, max_iter_, K, mae, mse, assign_wc_time, 
            update_wc_time, g_tot_wc_time, folder_);
#endif
    }
#ifdef DEBUG_INFO
    output_labels(n_, k, labels_, folder_);
    output_centers(k, seedset_, seedpos_, folder_);
#endif
    free();
    g_tot_wc_time  = omp_get_wtime() - start_wc_time;
    g_iter_wc_time = (g_tot_wc_time  - g_init_wc_time)  / max_iter_;
    
    return 0;
}

} // end namespace clustering
