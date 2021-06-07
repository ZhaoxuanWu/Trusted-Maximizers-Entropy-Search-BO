#! /bin/bash

TIME_STAMP=$(date +%m-%d)
FUNCTION=cnn\_cifar\_10
NUMQUERIES=10
NUMRUNS=2
NMAX=20
BATCHSIZE=20
NUMHYPS=1
NFEATURE=300
NPARAL=10
NSTO=10
NTRAIN=300
NYSAMPLE=100
NINIT=10
DTYPE=float64
FOLDER=$FUNCTION/result\_batch\_BO/$TIME_STAMP\_batch\_$BATCHSIZE\_run\_$NUMRUNS


if [[ ! -e $FOLDER ]]; then
    mkdir -p $FOLDER
fi

declare -a crit=('ftl')
# 'sftl')
declare -a mode=('empirical')
# 'sample')
declare -a gpu=(0)


for ((i=0;i<${#crit[@]};++i));
do
# Could add "nohup"
   python bo\_batch.py \
   --gpu "${gpu[i]}" \
   --function $FUNCTION \
   --folder "$FOLDER" \
   --criterion "${crit[i]}" \
   --mode "${mode[i]}" \
   --numqueries $NUMQUERIES \
   --numruns $NUMRUNS \
   --numhyps $NUMHYPS \
   --nmax $NMAX \
   --nfeature $NFEATURE \
   --nparal $NPARAL \
   --nsto $NSTO \
   --ntrain $NTRAIN \
   --nysample $NYSAMPLE \
   --ninit $NINIT \
   --batchsize $BATCHSIZE \
   --dtype $DTYPE \
#    > $FOLDER/log_"${crit[i]}".txt \
#    2> $FOLDER/err_"${crit[i]}".txt &
done

