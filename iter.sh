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

ind2f[3]="ego-net0"
ind2Ratings[3]="data/ego-net/0.edges.u2u.app"
ind2Attrs[3]="data/ego-net/0.circles.u2f.filtered"
ind2AttrNum[3]=34

ind2f[4]="ego-net107"
ind2Ratings[4]="data/ego-net/107.edges.u2u.app"
ind2Attrs[4]="data/ego-net/107.circles.u2f.filtered"
ind2AttrNum[4]=18

ind2f[5]="ego-net1684"
ind2Ratings[5]="data/ego-net/1684.edges.u2u.app"
ind2Attrs[5]="data/ego-net/1684.circles.u2f.filtered"
ind2AttrNum[5]=34

ind2f[6]="ego-net1912"
ind2Ratings[6]="data/ego-net/1912.edges.u2u.app"
ind2Attrs[6]="data/ego-net/1912.circles.u2f.filtered"
ind2AttrNum[6]=34

ind2f[7]="ego-net3437"
ind2Ratings[7]="data/ego-net/3437.edges.u2u.app"
ind2Attrs[7]="data/ego-net/3437.circles.u2f.filtered"
ind2AttrNum[7]=34

ind2f[8]="ego-net348"
ind2Ratings[8]="data/ego-net/348.edges.u2u.app"
ind2Attrs[8]="data/ego-net/348.circles.u2f.filtered"
ind2AttrNum[8]=28

ind2f[9]="ego-net3980"
ind2Ratings[9]="data/ego-net/3980.edges.u2u.app"
ind2Attrs[9]="data/ego-net/3980.circles.u2f.filtered"
ind2AttrNum[9]=34

ind2f[10]="ego-net414"
ind2Ratings[10]="data/ego-net/414.edges.u2u.app"
ind2Attrs[10]="data/ego-net/414.circles.u2f.filtered"
ind2AttrNum[10]=14

ind2f[11]="ego-net686"
ind2Ratings[11]="data/ego-net/686.edges.u2u.app"
ind2Attrs[11]="data/ego-net/686.circles.u2f.filtered"
ind2AttrNum[11]=28

ind2f[12]="ego-net698"
ind2Ratings[12]="data/ego-net/698.edges.u2u.app"
ind2Attrs[12]="data/ego-net/698.circles.u2f.filtered"
ind2AttrNum[12]=26


for ind in {3..12}                  # <-- N dataset
do
    f=${ind2f[${ind}]}
    for KNum in 100 200 300 400   # <-- 4 SVD_K_NUM
    do
        for iterNum in {0..4}       # vvv 10 iterations
        do
            echo "[info] Now f=${f} KNum=${KNum} iterNum==${iterNum}"
            let iter="${iterNum}*2"
            LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=${ind2AttrNum[${ind}]} SVD_K_NUM=${KNum} MAX_TRAIN_NM=10000000 python2.7 -u run.py ${ind2Ratings[${ind}]} ${ind2Attrs[${ind}]} ${iter} > report/${KNum}F${f}_${iter} &
            let iter2="${iterNum}*2+1"
            LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=${ind2AttrNum[${ind}]} SVD_K_NUM=${KNum} MAX_TRAIN_NM=10000000 python2.7 -u run.py ${ind2Ratings[${ind}]} ${ind2Attrs[${ind}]} ${iter2} > report/${KNum}F${f}_${iter2}
        done
                                  # ^^^ 10 iterations
    done
done

