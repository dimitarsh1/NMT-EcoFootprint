#!/bin/bash

i=$1

sed -r 's/^ //g' $i | sed -r 's/\s+/\t/g' | sed -r 's/\t+/\t/g' > $i.tab;

head -2 $i.tab > tmp.header.txt
sed '/^#/d' $i.tab > $i.clean.tmp;
cat tmp.header.txt $i.clean.tmp > $i.clean;

rm tmp.header.txt $i.clean.tmp 

echo "Done"
