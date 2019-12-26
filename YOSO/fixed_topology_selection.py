import time
from train import main 
start = time.time()
node = [2,4,8,16,32,64]
for i in node:
    for j in node:
        for q in node:
            main(i,j,q)
end = time.time()
print(end-start)
