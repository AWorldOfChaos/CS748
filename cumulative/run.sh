#!/bin/bash

echo 'QRM1'
python3 plot.py -H 1000,2000,5000,10000,15000,25000 -A qrm1

echo 'QRM2'
python3 plot.py -H 1000,2000,5000,10000,15000,25000 -A qrm2

echo 'Thompson'
python3 plot.py -H 1000,2000,5000,10000,15000,25000 -A thompson