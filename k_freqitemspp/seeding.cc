#include "seeding.h"

namespace clustering {

// -----------------------------------------------------------------------------
u64 labels_to_index(                // convert labels into index and index_pos
    int n,                              // number of labels
    int k,                              // number of centers
    int *labels,                        // data labels (allow modify)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos)           // bin position (return)
{
    // sort all labels and its corresponding index
    int *index = new int[n];
    int i = 0;
    std::iota(index, index+n, i++);
    std::sort(index, index+n, [&](int i,int j){return labels[i] < labels[j];});
    std::sort(labels, labels+n);
    
    // update binset and binpos
    binset.reserve(n);
    binset.insert(binset.end(), index, index+n);
    
    binpos.reserve(k+1); // reserve by the last number of centers
    binpos.push_back(0UL);
    for (int i = 1; i < n; ++i) {
        if (labels[i] != labels[i-1]) binpos.push_back(i);
    }
    binpos.push_back(n);
    
    delete[] index;
    return binpos.size()-1;
}

// -----------------------------------------------------------------------------
int labels_to_bins(                 // labels to bins & re-number labels
    int n,                              // number of data points
    int k,                              // number of cluster centers
    int *labels,                        // cluster labels for data (return)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos)           // bin position (return)
{
    std::vector<int>().swap(binset);
    std::vector<u64>().swap(binpos);
    
    // convert labels on local data into global bin set and bin position
    u64 num_bins = labels_to_index(n, k, labels, binset, binpos);
    assert(num_bins <= k && num_bins > 0);
    
    // re-number labels for local data
    int id = -1, num = -1;
    for (int i = 0; i < num_bins; ++i) {
        const int *bin = &binset[binpos[i]];    // get bin
        num = int(binpos[i+1] - binpos[i]); // get bin num
        
        for (int j = 0; j < num; ++j) { id = bin[j]; labels[id] = i; }
    }
    return num_bins;
}

} // end namespace clustering
