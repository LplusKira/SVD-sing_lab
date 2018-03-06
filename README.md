# how to run:
  ```
  pip install -r requirements.txt
  ```
  
  ```
  on ego-net:
  LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=18 SVD_K_NUM=100 MAX_TRAIN_NUM=10000000 python2.7 -u run.py data/ego-net/107.edges.u2u.app data/ego-net/107.circles.u2f.filtered 1 > report/100Fego-net107_0 &

  on yelp:
  LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=12 SVD_K_NUM=100 MAX_TRAIN_NUM=10000000 nohup python -u run.py data/yelp/yelp.data.filtered.int data/yelp/biz.cates.filtered.int 0 > report/100Fyelp_0 &
  # 																	                   ^ i.e. no buffer stdout
  #^ the bigger LAMBDA is, the more regularization in soft-max logistic regression  

  on 100k:
  LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=27 SVD_K_NUM=100 MAX_TRAIN_NUM=10000000 nohup python -u run.py data/ml-100k/u.data.filtered.sorted data/ml-100k/usr.cates.filtered 0 > report/100F100k_0 &

  on 1M:
  LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=30 SVD_K_NUM=100 MAX_TRAIN_NUM=10000000 nohup python -u run.py data/ml-1m/filtered.dat data/ml-1m/usr.cates.filtered 0 > report/100F1m_0 &
  ```

  ```
  ## gnuplot is required
  run cmd.sh ## e.g. ./cmd.sh train 1err
  ```

  ```
  cheat sheet in tmux (cause you dont need to nohup + '&' to keep the python ps running in background)
  on 100k:
  LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=27 SVD_K_NUM=100 MAX_TRAIN_NUM=10000000 python2.7 -u run.py data/ml-100k/u.data.filtered.sorted data/ml-100k/usr.cates.filtered 0 > report/100F100k_0

  on yelp:
  LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=12 SVD_K_NUM=10 MAX_TRAIN_NUM=10000000 python2.7 -u run.py data/yelp/yelp.data.filtered.int data/yelp/biz.cates.filtered.int 0 > report/100Fyelp_0

  on 1M:
  LAMBDA=0.001 USR_TOTAL_LABELS_FIELDS=30 SVD_K_NUM=100 MAX_TRAIN_NUM=10000000 python2.7 -u run.py data/ml-1m/filtered.dat data/ml-1m/usr.cates.filtered 0
  ```

# visualize results:
  modify $dataResources $line in examine.sh
  sh examine.sh
  

# data format:
  python -u run.py ratingFile attrFile randSeed

  ratingFile (1st arg after run.py) should be:
    1       114     5       875072173 
     ^ i.e. seperated by '\t'
    ^ usrid ^ itemid^ rating^ doesnt matter ... 

  attrFile (2nd arg after run.py) should be:
    10,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
    ^usrid
      ^ seperated by comma
       ^ concatenated one-hot encoded attributes 
    
  randSeed
    an integer (should best be the same as the suffix of redirected file)

# refs:
svd usage, ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
numpy loading txt file usage, ref: http://akuederle.com/stop-using-numpy-loadtxt
pandas: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

# TL;DR:
```
pip install -r requirements.txt
cd bin/; bash pre.sh; bash cmd.sh data; cd -;
cd SVD_sing/; \
SVD_K_NUM=10 MAX_TRAIN_NUM=10 \
LEARNING_RATE=0.001 LAMBDA=0.001 \
python svd_sing.py 0 2 ml-100k;
cd -;
```
