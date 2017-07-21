#!/bin/bash

mkdir Data
cd Data

#getting SP500 data
git clone https://github.com/clee1994/SP500
cd SP500
git clone https://github.com/c0redumb/yahoo_quote_download
python3 main.py
cd ..

#getting Reuters data
git clone https://github.com/clee1994/Reuters-full-data-set
cd Reuters-full-data-set
python3 generate.py
mv output* output
python3 topickle.py
cd ../..
