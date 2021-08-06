#!/bin/bash

for k in LSTM TRANS
do 
	for i in EN FR ES
	do 
		for j in EN FR ES
		do 
			if [ ! $i = $j ] && ( [ $j = EN ] || [ $i = EN ] )
			then 
				echo $i $j $k
				DIRNAME=${i}-${j}-${k}-BPE
				mkdir -p $DIRNAME
				cp /home/shterion/EcoNMT/engines/$DIRNAME/model/*.log $DIRNAME/
				cp /home/shterion/EcoNMT/engines/$DIRNAME/model/power_log* $DIRNAME/ -R
			fi
		done
	done
done
