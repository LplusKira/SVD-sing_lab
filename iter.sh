#!/bin/bash
# goal: iterate all dataset by 5 runs on 4 SVD_K_NUM

ind2f[0]="100k"
ind2Ratings[0]="data/ml-100k/u.data.filtered.sorted"
ind2Attrs[0]="data/ml-100k/usr.cates.filtered"
ind2AttrNum[0]=27 

ind2f[1]="1m"
ind2Ratings[1]="data/ml-1m/filtered.dat"
ind2Attrs[1]="data/ml-1m/usr.cates.filtered"
ind2AttrNum[1]=30

ind2f[2]="yelp"
ind2Ratings[2]="data/yelp/yelp.data.filtered.int"
ind2Attrs[2]="data/yelp/biz.cates.filtered.int"
ind2AttrNum[2]=12

ind2f[3]="ego-net"
ind2Ratings[2]="data/ego-net/0.edges.u2u.app"
ind2Attrs[2]="data/ego-net/0.circles.u2f.filtered"
ind2AttrNum[2]=34

ind2f[4]="ego-net"
ind2Ratings[2]="data/ego-net/107.edges.u2u.app"
ind2Attrs[2]="data/ego-net/107.circles.u2f.filtered"
ind2AttrNum[2]=18

ind2f[5]="ego-net"
ind2Ratings[2]="data/ego-net/1684.edges.u2u.app"
ind2Attrs[2]="data/ego-net/1684.circles.u2f.filtered"
ind2AttrNum[2]=34

ind2f[6]="ego-net"
ind2Ratings[2]="data/ego-net/1912.edges.u2u.app"
ind2Attrs[2]="data/ego-net/1912.circles.u2f.filtered"
ind2AttrNum[2]=34

ind2f[7]="ego-net"
ind2Ratings[2]="data/ego-net/3437.edges.u2u.app"
ind2Attrs[2]="data/ego-net/3437.circles.u2f.filtered"
ind2AttrNum[2]=34

ind2f[8]="ego-net"
ind2Ratings[2]="data/ego-net/348.edges.u2u.app"
ind2Attrs[2]="data/ego-net/348.circles.u2f.filtered"
ind2AttrNum[2]=28

ind2f[9]="ego-net"
ind2Ratings[2]="data/ego-net/3980.edges.u2u.app"
ind2Attrs[2]="data/ego-net/3980.circles.u2f.filtered"
ind2AttrNum[2]=34

ind2f[10]="ego-net"
ind2Ratings[2]="data/ego-net/414.edges.u2u.app"
ind2Attrs[2]="data/ego-net/414.circles.u2f.filtered"
ind2AttrNum[2]=14

ind2f[11]="ego-net"
ind2Ratings[2]="data/ego-net/686.edges.u2u.app"
ind2Attrs[2]="data/ego-net/686.circles.u2f.filtered"
ind2AttrNum[2]=28

ind2f[12]="ego-net"
ind2Ratings[2]="data/ego-net/698.edges.u2u.app"
ind2Attrs[2]="data/ego-net/698.circles.u2f.filtered"
ind2AttrNum[2]=26


for ind in 0 1 2                  # <-- 3 dataset
do
    f=${ind2f[${ind}]}
    for KNum in 100 200 300 400   # <-- 4 SVD_K_NUM
    do
        for iter in 0 1 2 3       # vvv 5 iterations
        do
            LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=${ind2AttrNum[${ind}]} SVD_K_NUM=${KNum} MAX_TRAIN_NM=10000000 python2.7 -u run.py ${ind2Ratings[${ind}]} ${ind2Attrs[${ind}]} ${iter} > report/${KNum}F${f}_${iter} &
        done
        LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=${ind2AttrNum[${ind}]} SVD_K_NUM=${KNum} MAX_TRAIN_NM=10000000 python2.7 -u run.py ${ind2Ratings[${ind}]} ${ind2Attrs[${ind}]} 4 > report/${KNum}F${f}_4 
                                  # ^^^ 5 iterations
    done
done

