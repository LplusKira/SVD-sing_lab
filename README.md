# Install:
```
pip install -r requirements.txt
cd bin/; bash cmd.sh data; cd -;
```

# Run:
- MovieLens100K, 5-fold cv
```
cd SVD_sing/; \
SVD_K_NUM=100 MAX_TRAIN_NUM=10000 LEARNING_RATE=0.001 LAMBDA=0.001 \
nohup python -u svd_sing.py 5 2 ml-100k;
cd -;
```

# Examine:
- Render report & raw stats in .csv
__cd bin/; bash pre.sh; cd -__ if it's not executed yet
```
cd bin/; \
bash genStats.sh ml-100k|ml-1m|youtube|ego-net348 foldNum; \
cd -;
```

# Structure:
- For model, please refer to [Your Cart tells You: Inferring Demographic Attributes from Purchase Data](https://github.com/LplusKira/SVD-sing_lab/blob/master/doc/WSDM2016_wang.pdf)'s section 4.1.2
- For codes' architecture
```
  loads data and keeps respective dependencies (by dataloader & pandas api)
  |
  V
  parse rating data to sparse matrix (through scipy api)
  |
  V
  solve SVD (through scipy api)
  |
  V
  by attribute, learn logistic regression (through sklearn api)
```

# In/Out format:
Follow the format described in [SNE's README.md](https://github.com/LplusKira/SNE_lab/blob/master/README.md)

# Ref:
0. [Dataset](http://files.grouplens.org/datasets/movielens/ml-100k.zip) for ml-100k
1. [Dataset](http://files.grouplens.org/datasets/movielens/ml-1m.zip) for ml-1m
2. [Dataset](http://snap.stanford.edu/data/facebook.tar.gz) for ego-net
3. [Graph Data](http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz) and [Community Data](http://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz) for youtube
4. [SVD APIs in scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)
5. [CSV loading APIs in Pandas](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)

# TODO:
- 'XXX' in files

# TL;DR:
- Try this simple one first: Will generate __10Fml-100k__ under report/
```
pip install -r requirements.txt
cd bin/; bash pre.sh; bash cmd.sh data; cd -;
cd SVD_sing/; \
SVD_K_NUM=10 MAX_TRAIN_NUM=10 \
LEARNING_RATE=0.001 LAMBDA=0.001 \
python svd_sing.py 0 2 ml-100k;
cd -;
```
