# k-FreqItems++

Welcome to the **k-FreqItems++** GitHub!

k-FreqItems++ is a sparse data clustering method based on Jaccard distance. This repo contains a single machine version of k-FreqItems++ with OpenMP optimization. There are three characteristics of k-FreqItems++ for this version:

- Friendly to the laptop with Ubuntu and easy-to-compile
- Auto-detect and leverage all CPU threads for parallel computation
- Parameter-light (only need to set $k$ and $\alpha$)


## Data Sets

### Data Sets Details

We have enclosed two small sparse data sets Amazon and News20 as toy examples for testing.
Users can also consider other sparse data sets with ground truth laebls for performance evaluation, i.e., [RCV1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#rcv1.multiclass), [URL](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#url), [Avazu](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#avazu), [KDD2012](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2012), [Criteo10M and Criteo1B](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#criteo_tb)). Users can download the data sets [here](https://drive.google.com/drive/folders/1UceZI0xjBC7WQTmzGVOF4DGDBq481tPx?usp=sharing). 

Since this repo consider a single machine version without using GPU, users should employ the data from `XXX/1/` folder, e.g., `URL/1/` for the URL dataset. Once you have downloaded the datasets, please put the data into the `data/1/` folder (or change to another directory by changing the data path in the bash script simultaneously).

The statistics of data sets are summarized as follows.

| Data Sets | # Data            | # Dim             | # Non-Zero Dim | Data Size | Global $\alpha$ | Local $\alpha$ |
| --------- | ----------------- | ----------------- | -------------- | --------- | --------------- | -------------- |
| News20    | $2.0 \times 10^4$ | $6.2 \times 10^4$ | 80             | 6.3 MB    | 0.2             | -              |
| RCV1      | $5.3 \times 10^5$ | $4.7 \times 10^4$ | 65             | 136 MB    | 0.2             | -              |
| URL       | $2.3 \times 10^6$ | $2.3 \times 10^6$ | 116            | 1.1 GB    | 0.4             | 0.2            |
| Criteo10M | $1.0 \times 10^7$ | $1.0 \times 10^6$ | 39             | 1.6 GB    | 0.2             | 0.1            |
| Avazu     | $4.0 \times 10^7$ | $1.0 \times 10^6$ | 15             | 2.6 GB    | 0.3             | 0.2            |
| KDD2012   | $1.5 \times 10^8$ | $5.4 \times 10^7$ | 11             | 7.3 GB    | 0.5             | 0.2            |
| Criteo1B  | $1.0 \times 10^9$ | $1.0 \times 10^6$ | 39             | 153 GB    | 0.2             | 0.1            |

### Data Format

The input sparse data sets are stored in binary format, where each file consists of two fields: `pos` and `data`, as shown below:

| field | field type       | description                             |
| ----- | ---------------- | --------------------------------------- |
| pos   | `uint64_t`*(n+1) | start position of `data` for each point |
| data  | `int32`*pos[n]   | non-zero dimensions IDs of all points   |

`pos` is an `uint64_t` array of (n+1) length, which stores the start position of the `data` array for each sparse data point. `data` is an `int32` array of pos[n] length, which store the non-zero dimensions IDs of all sparse data points. Here we assume that all non-zero dimension IDs can be represented by an `int32` type integer. If the dimensionality exceeds the range of `int32`, one can store the non-zero dimension IDs by `uint64_t` type with minor modification. With `pos` and `data`, one can efficient retrieve a specific data point with its data ID.

For example, suppose there is a sparse data set with four points: `x_0={1,3,5,8}`, `x_1={1,3}`, `x_2={1,6,8}`, and `x_3={1,8,10}`. Then, the `pos` array is `[0,4,6,9,12]`, e.g., pos[0]=0, pos[4]=12. And the `data` array is `[1,3,5,8,1,3,1,6,8,1,8,10]`. If you want to retrieve `x_1`, you can first get its start position of `data` and its length from `pos` by its data ID `1`, i.e., start position is `pos[1]=4`, and its length is `pos[1+1]-pos[1]=6-4=2`. Then you can retrieve `x_1` from `data` by the start position `4` and its length `2`, i.e., `x_1={1,3}`.

## Compilation

The source codes only require `g++` with `C++11` support. We have provided `Makefile` for compilation. Users can use the following commands to compile the source codes:

```bash
cd k_freqitemspp/
make clean
make -j
```

## Running k-FreqItems++

We have provided bash scripts to run k-FreqItems++. Users can set up different k values by simply running the following command:

```bash
cd k_freqitemspp/
./run_news20.sh   # the results can be found in `results/News20/`
./run_amazon.sh   # the results can be found in `results/Amazon/`
```

## Parameter Settings

In this repo, there are 6 parameters for k-FreqItems++, i.e., `n` (cardinality of data sets), `k` (number of pre-specified clusters), `alpha` (threshold for cluster center), `f` (data set format, `int32` by default), `dset` (address of data set), and `ofolder` (a folder to store output results).

### The Setting of $k$

Basically, once the data set is speficifed, `n`, `f`, `dset`, and `ofolder` are specified. Users only require setting `k` and `alpha`. For `k`, the valid range is $[1,n]$. Users can set up different `k` to watch the convergence of k-FreqItems++. The elbow point is often considered a suitable value of `k`. We will talk about `alpha` in next subsection.

### The Setting of $\alpha$

It should be noted that users need to set up an extra parameter called $\alpha$ ($0<\alpha<1$). Basically, different data sets have differnet optimal $\alpha$, and this value is data-dependent, which cannot derive based on closed-form formula. Nonetheless, based on our observations on more than 10 different data sets, the range of $\alpha \in [0.2, 0.5]$ can usually lead to a good result. By default, we suggest setting $\alpha=0.2$ or $\alpha=0.3$.

Besides, we have provided scripts together with the running scripts to illustrate how to tune this parameter (e.g., `run_amazon.sh` and `run_news20.sh`). If users plan to achieve near-optimal results, they can run those scripts for setting a near-optimal $\alpha$.

Thank you for your interests. It is welcome to contact me (huangq@comp.nus.edu.sg) if you meet any issue.

## Reference

Thank you so much for being so patient to read the user manual. We will appreciate using the following BibTeX to cite this work when you use k-FreqItemspp in your paper.

```tex
@article{huang2023new,
  title={A New Sparse Data Clustering Method Based On Frequent Items},
  author={Huang, Qiang and Luo, Pingyi and Tung, Anthony KH},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={1},
  pages={1--28},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

It is welcome to contact me (<huangq@comp.nus.edu.sg>) if you meet any issue. Thank you.
