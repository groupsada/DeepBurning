#!/usr/bin/python
import sys, math,random
import train

Alpha = 0.9
Beta  = 0.09
Gamma = 0.01

for i in range(len(sys.argv)-1):  
    if (sys.argv[i] == '-Alpha'):
        Alpha = float(sys.argv[i+1])
    elif(sys.argv[i] == '-Beta'):
        Beta = float(sys.argv[i+1])
    elif(sys.argv[i] == '-Gamma'):
        Gamma = float(sys.argv[i+1])

yValue,npu_cost,npu_time = train.main(HIDDEN_NODE_FC1, HIDDEN_NODE_FC2, HIDDEN_NODE_FC3)
speedUp = 1300./(float(npu_time))
energy  = 1e6/(float(npu_cost))
final_score = -yValue*Alpha - speedUp*Beta - Gamma*energy
print("Result of algorithm run: SUCCESS, 0, 0, %f, 0"%final_score)
