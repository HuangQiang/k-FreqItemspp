#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  Basic parameters
# ------------------------------------------------------------------------------
dname=News20                        # data set name
n=19928                             # cardinality
format=int32                        # data format: uint16, int32
dset=../data/1/${dname}.bin         # address of data set
ofolder=results/${dname}/           # output folder

# k=100
# for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# do 
#   ./kpp -n ${n} -k ${k} -a ${alpha} -f ${format} -ds ${dset} -of ${ofolder}
# done 

alpha=0.2
for k in 20 40 60 80 100 120 140 160 180 200
do
  ./kpp -n ${n} -k ${k} -a ${alpha} -f ${format} -ds ${dset} -of ${ofolder}
done
