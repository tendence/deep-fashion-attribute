#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:35:25 2018

@author: jack
"""

from config import cfg
import os
from torch.autograd import Variable

mycount=[]
for i in range(1000):
    mycount.append(0)
list_category_img = os.path.join(cfg.DATASET_BASE, r'Anno', r'list_attr_img_sub.txt')
with open(list_category_img) as fin:
    lines = fin.readlines()[2:]
    lines = list(filter(lambda x: len(x) > 0, lines))
    lines = list(map(lambda x: x.strip().split(), lines))
    for line in lines:
        name=line[0]
        attr=list(map(lambda x:int(x),line[1:]))
        attr=Variable(attr)
        if len(attr)!=1000:
            print('sv')
        print (name)
    #for i in range(1,1001):
     #   numbers=list(filter(lambda x:int(x.strip().split()[i]),lines))
      #  mycount.append(numbers.count(1))
#    j=0
#    for line in lines:
#        j=j+1
#        if j%1000==0:
#            print(j)
#        for i in range(1000):
#            if line[i+1]=='1':    
#                mycount[i]+= 1 
    
            
#output = os.path.join(cfg.DATASET_BASE, r'Anno', r'attribut_analytic.txt')
#with open(output, "w") as fw:
#    for i in range(len(mycount)):
#        fw.write("%d %d\n" % (i, mycount[i]))
        
#df=pd.read_csv(list_category_img,skiprows=1,delim_whitespace=True)