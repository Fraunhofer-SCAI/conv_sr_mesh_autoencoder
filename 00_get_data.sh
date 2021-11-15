#!/bin/bash

dataset=$1

echo "Download data for dataset: " $dataset

mkdir -p data
cd data

if [[ "$dataset" == "car_TRUCK" ]]
then
	wget https://owncloud.scai.fraunhofer.de/index.php/s/oRsS9poCSewArd5/download/car_TRUCK.tar.gz
elif [[ "$dataset" == "car_YARIS" ]]
then 
	wget https://owncloud.scai.fraunhofer.de/index.php/s/wP9pBCH7zdMY4sC/download/car_YARIS.tar.gz
elif [[ "$dataset" == "FAUST" ]]
then 
	wget https://owncloud.scai.fraunhofer.de/index.php/s/e3p6sWjzFccJDnt/download/FAUST.tar.gz
elif [[ "$dataset" == "gallop" ]]
then 
	wget https://owncloud.scai.fraunhofer.de/index.php/s/5FZ8t6AoyaKpemF/download/gallop.tar.gz
else
	echo "Not a valid dataset."
fi


echo "tar -xzvf $dataset.tar.gz && rm $dataset.tar.gz"
tar -xzvf $dataset.tar.gz && rm $dataset.tar.gz
echo "downloaded the data and putting it in: data/$dataset"
