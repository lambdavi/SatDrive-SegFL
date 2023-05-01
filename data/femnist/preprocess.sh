#!/usr/bin/env bash

# download data and convert to .json format

if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    cd preprocess
    ./data_to_json.sh
    cd ..
fi

NAME="femnist" # name of the dataset, equivalent to directory name

cd ./utils

./preprocess.sh --name $NAME $@

SAMPLING="$2"
echo "Sampling " ${SAMPLING}
cd ../data

mkdir ${SAMPLING}
mkdir ${SAMPLING}/sampled_data
mv sampled_data/* ${SAMPLING}/sampled_data/
rm -rf sampled_data
mkdir ${SAMPLING}/rem_user_data
mv rem_user_data/* ${SAMPLING}/rem_user_data/
rm -rf rem_user_data
mkdir ${SAMPLING}/train ${SAMPLING}/test
mv test/* ${SAMPLING}/test
mv train/* ${SAMPLING}/train
rm -rf train
rm -rf test