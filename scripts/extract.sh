#!/bin/bash
zip -FF ../raw_data/AIC21_Track5_NL_Retrieval.zip --out ../raw_data/AIC21_Track5_NL_Retrieval_full.zip
rm ../raw_data/AIC21_Track5_NL_Retrieval.z*
mkdir ../raw_data/AIC21_Track5_NL_Retrieval
unzip -FF ../raw_data/AIC21_Track5_NL_Retrieval_full.zip -d ../raw_data/