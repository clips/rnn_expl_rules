#!/usr/bin/env bash

dir_in=/home/madhumita/sepsis_mimiciii/text/
dir_out=/home/madhumita/sepsis_mimiciii/tokenized/

test -d $dir_out || mkdir $dir_out

for file in ls "$dir_in"*.txt
do
#echo "Tokenizing $file"
fname=$(basename $file)
ucto -L eng -n -l $file $dir_out$fname
done

rm -r $dir_in
mv $dir_out $dir_in