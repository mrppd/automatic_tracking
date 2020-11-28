#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:30:35 2020

@author: pronaya
"""

import pickle as pkl
import os
import numpy as np
import pandas as pd

df_interaction = ""
main_path = "/home/pronaya/Videos/Work/Educational info/Gottingen/Thesis"
with open(os.path.join(main_path, 'data/df_interaction.pickle'), 'rb') as handle:
    df_interaction = pkl.load(handle)
    
df_interaction.to_csv(os.path.join(main_path, 'data/df_interaction.csv'))

df_tmp = pd.DataFrame(columns=df_interaction.columns)

rows_list = []
for row in df_interaction:
    rows_list.append(row)
    
df_tmp = pd.DataFrame(rows_list)          