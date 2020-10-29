#!/usr/bin/env bash

while getopts i:o: flag
do
    case "${flag}" in
        i) dir_in=${OPTARG};;
        o) dir_out=${OPTARG};;
    esac
done

echo $dir_in
echo $dir_out

test -d $dir_out || mkdir $dir_out

for file in ls "$dir_in"*.txt
do
#echo "Tokenizing $file"
fname=$(basename $file)
ucto -L eng -n -l $file $dir_out$fname
done

rm -r $dir_in
mv $dir_out $dir_in