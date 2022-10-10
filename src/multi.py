from multiprocessing import Pool, cpu_count
import os
import psutil

cpu_count1 = cpu_count()
cpu_count2 = len(psutil.Process().cpu_affinity())
cpu_count3 = psutil.cpu_count()
cpu_count4 = os.cpu_count()
print(f"cpu count1: {cpu_count1}")
print(f"cpu count2: {cpu_count2}")
print(f"cpu count3: {cpu_count3}")
print(f"cpu count4: {cpu_count4}")

def f(x):
    return x*x

with Pool(cpu_count1) as p:
    print(p.map(f, [1, 2, 3]))