#!/bin/bash

dataResource="yelp" #1m"
run=900
line=80

echo "at run == ${run}"
for f in "100F" "200F" "300F" "400F"
do
  echo ""
  echo "$f"
  echo "microF1"
  grep "run ==  ${run}" -A ${line} report/${f}${dataResource}_* | grep "valid data microF1" | awk -F'==' '{print $2}'
  echo "one"
  grep "run ==  ${run}" -A ${line} report/${f}${dataResource}_* | grep "valid data oneError" | awk -F'==' '{print $2}'
  echo "rl"
  grep "run ==  ${run}" -A ${line} report/${f}${dataResource}_* | grep "valid data RL" | awk -F'==' '{print $2}'
  echo "coverage"
  grep "run ==  ${run}" -A ${line} report/${f}${dataResource}_* | grep "valid data coverage" | awk -F'==' '{print $2}'
  echo "avg Prec"
  grep "run ==  ${run}" -A ${line} report/${f}${dataResource}_* | grep "valid data avgPrec" | awk -F'==' '{print $2}'
  echo "hamming"
  grep "run ==  ${run}" -A ${line} report/${f}${dataResource}_* | grep "valid data hammingLoss" | awk -F'==' '{print $2}'
  echo "its about time"
  grep -w "run ==  1" -A 2 report/${f}${dataResource}_* | grep "time" > tmpr
  grep -w "run ==  ${run}" -A 2 report/${f}${dataResource}_* | grep "time" >> tmpr 
  awk -F'==' '{a[$1] = a[$1]" "$2;}END{for(i in a){ print i""a[i]; }}' tmpr | sort -t" " -k1,2 | awk -F' ' '{print $3" "$4" ~ "$5" "$6;}'
  rm tmpr
done
