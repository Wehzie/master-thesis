from multiprocessing import Pool, cpu_count
import os
import psutil

def test_cpu_count_alignment(verbose: bool = False) -> None:
    """test that various methods of determining the number of available CPUs align"""
    cpu_count1 = cpu_count()
    cpu_count2 = len(psutil.Process().cpu_affinity())
    cpu_count3 = psutil.cpu_count()
    cpu_count4 = os.cpu_count()
    if verbose:
        print(f"cpu count1: {cpu_count1}")
        print(f"cpu count2: {cpu_count2}")
        print(f"cpu count3: {cpu_count3}")
        print(f"cpu count4: {cpu_count4}")

    assert cpu_count1 == cpu_count2 == cpu_count3 == cpu_count4, "CPU counts don't align, re-evaluate which count method to use"

def f(x):
    return x*x

def test_mp() -> None:
    """more of a documentation than a test that could fail"""

    with Pool(cpu_count()) as p:
        p.map(f, [1, 2, 3])