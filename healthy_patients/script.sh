#!/bin/bash

for input in 'Pt12' 'Pt3' 'Pt13' 'Pt4' 'Pt18' 'Pt8' 'Pt14' 'Pt15' 'Pt9' 'PtTetniak1' 'Pt19' 'Pt5' 'Pt6' 'Pt20' 'Pt16' 'Pt17' 'Pt7' 'Pt10' 'Pt11'
do
    echo " ====== Analizing file $input ======="
    python3 Fitting.py "$input"
done
