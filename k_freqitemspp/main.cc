#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdint.h>

#include "util.h"
#include "k_freqitems.h"

using namespace clustering;

// -----------------------------------------------------------------------------
void usage()                        // display the usage
{
    printf("\n"
        "--------------------------------------------------------------------\n"
        " Parameters of K-FreqItems++                                        \n"
        "--------------------------------------------------------------------\n"
        " -n  {integer}  number of data points in a data set\n"
        " -k  {integer}  number of clusters\n"
        " -m  {integer}  maximum iterations\n"
        " -GA {real}     global alpha\n"
        " -LA {real}     local  alpha\n"
        " -F  {string}   data format: uint16, int32\n"
        " -P  {string}   prefix of data set\n"
        " -O  {string}   output folder to store output files\n"
        "\n\n\n");
}

// -----------------------------------------------------------------------------
template<class DType>
void kfreqitems_impl(               // k-freqitems implementation
    int   n,                            // number of data points
    int   k,                            // number of clusters
    float alpha,                        // global alpha
    const char *addr_data,              // address of data set
    const char *folder)                 // output folder to store output files
{
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    // -------------------------------------------------------------------------
    //  read dataset & init k-freqitems++
    // -------------------------------------------------------------------------
    DType *dataset = nullptr;
    u64   *datapos = new u64[n+1];
    dataset = read_sparse_data<DType>(n, addr_data, datapos);

    FILE *fp = nullptr;
    char fname[100]; sprintf(fname, "%skFreqItems++.csv", folder);
    create_dir(fname);
    // fp = fopen(fname, "a+");
    // if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    // fprintf(fp, "K,MSE,MAE,WTime,k,MaxIter,Iter,");
    // fprintf(fp, "Alpha,InitWTime,IterWTime,TotWTime\n");
    // fclose(fp);
    
    KFreqItems<DType> *k_freqitems = new KFreqItems<DType>(n, MAX_ITER, alpha, 
        folder, (const DType*) dataset, (const u64*) datapos);
    
    // -------------------------------------------------------------------------
    //  k_freqitems: k-modes clustering for sparse data
    // -------------------------------------------------------------------------
    int ret = k_freqitems->clustering(k);
    
    if (ret == 0) {
        printf("K = %d, MSE = %f, MAE = %f, K-FreqItems++ = %.2lf Seconds\n", 
            g_k, g_mse, g_mae, g_kpp_wc_time);
        printf("Init = %.2lf Seconds\n", g_init_wc_time);
        printf("Iter = %.2lf Seconds\n", g_iter_wc_time);
        printf("Tot  = %.2lf Seconds\n", g_tot_wc_time);
        printf("\n");
        
        // write the results of each setting to disk
        fp = fopen(fname, "a+");
        if (!fp) { printf("ERROR: cannot open %s\n", fname); return; }
        
        fprintf(fp, "%d,%f,%f,%.2lf,", g_k, g_mse, g_mae, g_kpp_wc_time);
        fprintf(fp, "%d,%d,%d,%g,%.2lf,%.2lf,%.2lf\n", k, MAX_ITER, g_iter, 
            alpha, g_init_wc_time, g_iter_wc_time, g_tot_wc_time);
        fclose(fp);
    }
    delete   k_freqitems;
    delete[] dataset;
    delete[] datapos;
}


// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    srand(RANDOM_SEED);             // use a fixed random seed
    
    int   n     = -1;               // number of data points
    int   k     = -1;               // number of seeds
    float alpha = -1.0f;            // global \alpha
    char  format[20];               // data format: uint8,uint16,int32,float32
    char  addr_data[200];           // address of data set
    char  folder[200];              // output folder to store output files
    
    int cnt = 1;
    while (cnt < nargs) {
        if (strcmp(args[cnt], "-n") == 0) {
            n = atoi(args[++cnt]); assert(n > 0);
            printf("n=%d\n", n);
        }
        else if (strcmp(args[cnt], "-k") == 0) {
            k = atoi(args[++cnt]); assert(k > 0);
            printf("k=%d\n", k);
        }
        else if (strcmp(args[cnt], "-a") == 0) {
            alpha = atof(args[++cnt]); assert(alpha >= 0);
            printf("alpha=%g\n", alpha);
        }
        else if (strcmp(args[cnt], "-f") == 0) {
            strncpy(format, args[++cnt], sizeof(format));
            printf("format=%s\n", format);
        }
        else if (strcmp(args[cnt], "-ds") == 0) {
            strncpy(addr_data, args[++cnt], sizeof(addr_data));
            printf("addr_data=%s\n", addr_data);
        }
        else if (strcmp(args[cnt], "-of") == 0) {
            strncpy(folder, args[++cnt], sizeof(folder));
            int len = (int) strlen(folder);
            if (folder[len-1]!='/') { folder[len]='/'; folder[len+1]='\0'; }
            printf("folder=%s\n", folder);
        }
        else {
            printf("Parameters error!\n"); usage(); exit(1);
        }
        ++cnt;
    }
    // -------------------------------------------------------------------------
    //  methods 
    // -------------------------------------------------------------------------
    if (strcmp(format, "uint16") == 0) {
        kfreqitems_impl<u16>(n, k, alpha, addr_data, folder);
    }
    else if (strcmp(format, "int32") == 0) {
        kfreqitems_impl<int>(n, k, alpha, addr_data, folder);
    }
    else {
        printf("Parameters error!\n"); usage();
    }
    return 0;
}
