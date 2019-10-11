from subprocess import Popen
import os

arr = os.listdir()
arr.sort()

item_store = []
for item in arr:
   if(item != "gpu.py" and item != "nohup.out" and item !=  "edward.py"):
      item_store.append(item)

arr = item_store

o = 1

size = 10

for item in arr[0:1]:
   items = os.path.abspath(__file__)[0:-9] + item
   big = len(os.listdir(items))
   s = int(big/size + (1 - (big/size - int(big/size))))

   #for i in range(s):
   for i in range(s):
      if(big <= (i+1)*size):
         os.system("python gpu.py " + str(i*size) +" " + str(i*size + (size - ((i+1)*size - big) )) + " " + str(o))
      else:
         os.system("python gpu.py " + str(i*size) +" " + str((i+1)*size) + " " + str(o))
  
   o = o  + 1

#for i in range(2):
#  for j in range():

###Popen('python filename.py')
