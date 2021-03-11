#!/bin/bash

#copying files from 0..9 directory
#target directory
mkdir $1

for i in {0..9}
do
 echo "Copying files from $i"
 cp $i/*.png $1
done

echo "Total files in $1:"
ls $1 | wc -l 
