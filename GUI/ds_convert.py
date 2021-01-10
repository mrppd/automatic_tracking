# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 01:17:34 2021

@author: pronaya
"""

import sys
import pandas as pd
import os


source_dir = os.path.dirname(os.path.abspath(__file__))

print("\n")
print("####################################################")

if(len(sys.argv)==3):    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("Input file:", input_file)
    print("Output file:", output_file)
    
    metaDataTmp = pd.read_csv(input_file)

    metaDataTmp = metaDataTmp.astype(str)

    metaDataTmp.p1 = metaDataTmp.p1.str.replace(',',':')
    metaDataTmp.p2 = metaDataTmp.p1.str.replace(',',':')
    metaDataTmp.p3 = metaDataTmp.p1.str.replace(',',':')
    metaDataTmp.p4 = metaDataTmp.p1.str.replace(',',':')

    metaDataTmp.to_csv(output_file, sep=",", index=False)

    print("Dataset converted to the new format!")
else:
    print("Conversion unsuccessful!!!")

print("####################################################")



