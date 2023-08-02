#!/usr/bin/env bash

config=../config/config-bert.txt
train_data=../data/covid19/raw_train_all_onecol.csv
dev_data=../data/covid19/raw_val_all_onecol.csv
test_data=../data/covid19/raw_test_all_onecol.csv
out_dir=./model/

for seed in 4
do
    echo "Start training on seed ${seed}......"
    for dataset in "covid19"
    do
        echo "Start training on dataset ${dataset}......"
        python train_model.py -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -outdir ${out_dir} -dataset ${dataset} -lr1 2e-5 -lr2 1e-3 -d 0. -t 1 -s ${seed} -step 5 -gen 4 -clipgrad -anneal -calib
    done
done

config=../config/config-bert.txt
train_data=../data/argmin/raw_train_all_onecol.csv
dev_data=../data/argmin/raw_val_all_onecol.csv
test_data=../data/argmin/raw_test_all_onecol.csv
out_dir=./model/

for seed in 4
do
    echo "Start training on seed ${seed}......"
    for dataset in "argmin"
    do
        echo "Start training on dataset ${dataset}......"
        python train_model.py -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -outdir ${out_dir} -dataset ${dataset} -lr1 2e-5 -lr2 1e-3 -d 0. -t 1 -s ${seed} -step 5 -gen 4 -clipgrad -anneal -calib
    done
done

config=../config/config-bert.txt
train_data=../data/pstance/raw_train_all_onecol.csv
dev_data=../data/pstance/raw_val_all_onecol.csv
test_data=../data/pstance/raw_test_all_onecol.csv
out_dir=./model/

for seed in 4
do
    echo "Start training on seed ${seed}......"
    for dataset in "pstance"
    do
        echo "Start training on dataset ${dataset}......"
        python train_model.py -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -outdir ${out_dir} -dataset ${dataset} -lr1 2e-5 -lr2 1e-3 -d 0. -t 1 -s ${seed} -step 5 -gen 4 -clipgrad -anneal -calib
    done
done
