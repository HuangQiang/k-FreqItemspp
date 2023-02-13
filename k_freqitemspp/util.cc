#include "util.h"

namespace clustering {

int g_k            = -1;            // global param: #clusters
f32 g_mae          = -1.0f;         // global param: mean absolute error
f32 g_mse          = -1.0f;         // global param: mean square   error
f64 g_tot_wc_time  = -1.0;          // global param: total wall clock time (s)

int g_iter         = -1;            // global param: #iterations
f64 g_init_wc_time = -1.0;          // global param: init wall clock time (s)
f64 g_iter_wc_time = -1.0;          // global param: iter wall clock time (s)
f64 g_kpp_wc_time  = -1.0;          // global param: k-freqitems++ wall clock time (s)

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path does not exist
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] != '/') continue;
        
        char ch = path[i+1]; path[i+1] = '\0';
        if (access(path, F_OK) != 0) { // create the directory if not exist
            if (mkdir(path, 0755) != 0) {
                printf("Could not create %s\n", path); exit(1);
            }
        }
        path[i+1] = ch;
    }
}

// -----------------------------------------------------------------------------
float uniform(                      // gen a random variable from uniform distr.
    float start,                        // start position
    float end)                          // end position
{
    assert(start < end);
    return start + ((end-start)*rand() / (float) RAND_MAX);
}

} // end namespace clustering
