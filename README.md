# Circuit Generator

- build graphs in Python using networkX
- convert graphs to SPICE netlists 
- run SPICE netlists with ngspice
- solve non-linear transformation (NLT) tasks. for example sinus to square wave.


## DevOps

- Before running a full sweep run a test sweep with m-averages=1 with the final parameters
- Before running a full sweep run a test sweep with the test parameters on the target hardware

## IDEAS and TODOS

- use jupyter notebook to call most important functions and serve as bridge between paper and code
- why is signal generation interesting
    - mathematical literature
        - fourier transform for signal generation, which peeks to choose to get what accuracy?
        - to what extent is this an open problem?
        - analytical solution available?