###The script consists of code of synchronous and asynchronous equations .
###The code is implemented in parallel process. 

from __future__ import division
from numba import cuda
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import sys
import numba

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def gpu(files, place):
    m = len(files)
    #Stored the files into array
    output = []
    t = 1
    temp =[]
    for file in files:
        arr = []
        for line in file:
            if (len(line.split()) == 2 and isfloat(line.split()[0]) and isfloat(line.split()[1])):
                arr.append(float(line.split()[1]))
            if(len(line.split()) == 2 and t == 1 and isfloat(line.split()[0]) and isfloat(line.split()[1])):
                temp.append(float(line.split()[0]))
        t = 0
        output.append(arr)

    output = np.array(output)

    threadsperblock = (8, 8)
    blockspergrid = (452, 452)

    ############################################
    #Synchronous
    #summation of output array
    SUM = np.sum(output, axis=0)
    #division of array
    DIVIDE = np.divide(SUM, m)

    #A_tudor array
    A_tudor = []
    for file in output:
        A_tudor.append(np.subtract(file, DIVIDE))
    A_tudor = np.array(A_tudor)

    # CUDA kernel
    @cuda.jit
    def multi(A,C):
        row, col = cuda.grid(2)
        z = row*3601+col
        if(z < 3601*3601):
            C[z] = A[row]*A[col]

    multi_A_bar = []
    
    for i in range(m):
        A = np.array(A_tudor[i])
        A_global_mem = cuda.to_device(A)
        C_global_mem = cuda.device_array(len(A_tudor[0])*len(A_tudor[0]))
        multi[blockspergrid, threadsperblock](A_global_mem,  C_global_mem)
        C = C_global_mem.copy_to_host()
        multi_A_bar.append(C)
        
    multi_A_bar_sum = np.sum(multi_A_bar, axis = 0)
    synchronous = np.divide(multi_A_bar_sum, m-1)

    ####################################
    #Asynchronous

    # CUDA kernel
    @cuda.jit
    def multi2(A,C):
        row, col = cuda.grid(2)
        z = row*3601+col
        C[z] = 0
        if(z < 3601*3601 and C[z] == 0):
            Sum = 0
            for i in range(0,m):
                v = 0
                for k in range(0,m):
                    if(k == i):
                       Nik = 0
                       v = v + Nik*A[k][row]
                    else:   
                       Nik = 1/(np.pi*(k-i))
                       v = v + Nik*A[k][row]
                Sum = Sum + A[i][col]*v
                   
            C[z] = Sum/(m-1)


    #A = np.array(A_tudor,dtype = 'f')
    A = np.array(A_tudor)
    A_global_mem = cuda.to_device(A)
    C_global_mem = cuda.device_array(len(A_tudor[0])*len(A_tudor[0]))
    multi2[blockspergrid, threadsperblock](A_global_mem,  C_global_mem)
    C = C_global_mem.copy_to_host()
    asyn = C
    
    #################################
    
    asyn_arr = [[0 for _ in range(len(temp))] for _ in range(len(temp))]
    synchronous_arr = [[0 for _ in range(len(temp))] for _ in range(len(temp))]

    z = 0
    for o in range(len(temp)): 
        for p in range(len(temp)):
            asyn_arr[p][o] = asyn[z]   
            synchronous_arr[p][o] = synchronous[z]
            z = z + 1
   
    x = np.arange(0, 3601, 1)
    y = np.arange(0, 3601, 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.contour(X, Y, synchronous_arr)
    plt.savefig(place + "/" + "ts.png")
    
    asyn_arr = np.array(asyn_arr)
    x = np.arange(0, 3601, 1)
    y = np.arange(0, 3601, 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.contour(X, Y, asyn_arr)
    #plt.imshow(asyn_arr)
    plt.savefig(place + "/" + "ta.png")
    plt.close('all')
    
    ####################################
    """
    #Write synchronous and asynchronous file
    s = open(place+"/ts", "w")
    a = open(place+"/ta", "w")
   
    z = 0
    for o in range(len(temp)):
        for p in range(len(temp)):
            s.write(str(temp[o]) + "   " + str(temp[p]) + "   " + str(synchronous[z]))
            s.write("\n")
            a.write(str(temp[o]) + "   " + str(temp[p]) + "   " + str(asyn[z]))
            a.write("\n")
            z = z + 1
    """
    
            
#cuda.current_context().reset()

arr = os.listdir()
arr.sort()

item_store = []
for item in arr:
   if(item != "gpu.py" and item != "nohup.out" and item !=  "edward.py"):
      item_store.append(item)

arr = item_store

def gpu_file(xxx, yyy, zzz):
   total = 0
   files = []
   place = os.path.abspath(__file__)[0:-6]
   
   for item in arr[zzz-1:zzz]:
      items = os.listdir(place + item)
      items.sort()
      #print(items)
      for i in items[xxx:yyy]:
         p = os.listdir(place + item + "/" +i)
         for a in p:
            if("asc" in a):
               total = total + 1
               pl = place + item + "/" + i + "/" + a
               f = open(pl, "r")
               files.append(f)
            
         if(total > 0):
             print(place + item + "/" + i)
             gpu(files, place + item + "/" + i)
             files = []
             total = 0
    
if __name__== "__main__":
    
    #print(sys.argv[3])
    gpu_file(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    #print(0, 20, 1)



#numba.cuda.profile_stop()
#gpu_file(0, 20, 1)
#numba.cuda.profile_stop()
#export NUMBAPRO_NVVM=/usr/local/cuda-8.0/nvvm/lib64/libnvvm.so.3.1.0
#export NUMBAPRO_LIBDEVICE=/usr/local/cuda-8.0/nvvm/libdevice

