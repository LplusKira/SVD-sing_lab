# how to run:
  ```
  pip install -r requirements.txt
  ```
  
  ```
  on yelp:
  USR_TOTAL_LABELS_FIELDS=12 SVD_K_NUM=100 MAX_TRAIN_NUM=10000 nohup python -u run.py data/yelp/yelp.data.filtered.int data/yelp/biz.cates.filtered.int 0 > report/100Fyelp_0 &
  # 																	    ^ i.e. no buffer stdout

  on 100k:
  USR_TOTAL_LABELS_FIELDS=27 SVD_K_NUM=100 MAX_TRAIN_NUM=10000 nohup python -u run.py data/ml-100k/u.data.filtered.sorted data/ml-100k/usr.cates.filtered 0 > report/100F100k_0 &

  on 1M:
  USR_TOTAL_LABELS_FIELDS=30 SVD_K_NUM=100 MAX_TRAIN_NUM=10000 nohup python -u run.py data/ml-1m/filtered.dat data/ml-100k/usr.cates.filtered 0 > report/100F1m_0 &
  ```

  ```
  ## gnuplot is required
  run cmd.sh ## e.g. ./cmd.sh train 1err
  ```

# data format:
  ...

