#!/bin/bash

for input1 in 'Pt3' 'Pt4' 'Pt13' 'Pt29' 'Pt9' 'Pt17' 'Pt18' 'Pt32' 'Pt33' 'Pt20' 'Pt2' 'Pt31' 'Pt1' 'Pt24' 'Pt25' 'Pt27' 'Pt5' 'Pt6' 'Pt28' 'Pt14' 'Pt8' 'Pt12' 'Pt16' 'Pt10' 'Pt34' 'Pt35' 'Pt30' 'Pt22' 'Pt11' 'Pt7' 'Pt26' 'Pt21' 'Pt15' 'Pt19' 'Pt23'
do
    for input2 in 'ShortBreak' 'LongBreak'
    do 
        for input3 in 'BeforeStart' 'AfterStart' 'BeforeEnd' 'AfterEnd'
        do
            echo " ====== Analizing file $input1 $input2 $input3 ======="
            python3 FittingHD.py "$input1" "$input2" "$input3"
        done
    done
done

