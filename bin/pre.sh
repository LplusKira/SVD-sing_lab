#!/bin/bash

## Pulling necessary scripts from SNE's repo
echo "[info] Pulling necessary scripts from SNE's repo"
downloadURL="https://raw.githubusercontent.com/LplusKira/SNE_lab/master/bin/"
for item in cmd.sh ml-100k.sh ml-1m.sh ego-net.sh youtube.sh \
    fixML100KFeatures.py fixML1MFeatures.py fixENFeatures.py fixYoutubeFeatures.py \
    genStats.sh plot.sh utils.sh; do
    curl -s -O ${downloadURL}${item}
done
