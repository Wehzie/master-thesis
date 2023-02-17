from pathlib import Path
import pandas as pd

test_data = """

###############################################################################
Peregrine Cluster
Job 28603101 for user 's3258734'
Finished at: Tue Feb 14 18:05:10 CET 2023

Job details:
============

Job ID              : 28603101
Name                : python-amplitude
User                : s3258734
Partition           : short
Nodes               : pg-node196
Number of Nodes     : 1
Cores               : 1
Number of Tasks     : 1
State               : COMPLETED
Submit              : 2023-02-14T18:03:11
Start               : 2023-02-14T18:03:11
End                 : 2023-02-14T18:05:10
Reserved walltime   : 00:30:00
Used walltime       : 00:01:59
Used CPU time       : 00:01:46 (efficiency: 89.91%)
% User (Computation): 97.21%
% System (I/O)      :  2.79%
Mem reserved        : 2G
Max Mem (Node/step) : 105.98M (pg-node196, per node)
Full Max Mem usage  : 105.98M
Total Disk Read     : 32.26M
Total Disk Write    : 398.59K


Acknowledgements:
=================

Please see this page for information about acknowledging Peregrine in your publications:

https://wiki.hpc.rug.nl/peregrine/introduction/scientific_output

################################################################################

==> job-python-frequency-28603097.log <==

  0%|          | 0/699 [00:00<?, ?it/s]
100%|██████████| 699/699 [00:00<00:00, 8311.72it/s]
saved freq_range_around_vo2 results to data/quantitative_experiment_python_sweep0/freq_range_around_vo20

time elapsed for func 'invoke_python_generator_sweeps': 268.10 s


time elapsed for func 'main': 268.11 s



###############################################################################
Peregrine Cluster
Job 28603097 for user 's3258734'
Finished at: Tue Feb 14 18:07:56 CET 2023

Job details:
============

Job ID              : 28603097
Name                : python-frequency
User                : s3258734
Partition           : short
Nodes               : pg-node201
Number of Nodes     : 1
Cores               : 1
Number of Tasks     : 1
State               : COMPLETED
Submit              : 2023-02-14T18:03:11
Start               : 2023-02-14T18:03:11
End                 : 2023-02-14T18:07:56
Reserved walltime   : 00:30:00
Used walltime       : 00:04:45
Used CPU time       : 00:04:35 (efficiency: 96.78%)
% User (Computation): 99.12%
% System (I/O)      :  0.88%
Mem reserved        : 2G
Max Mem (Node/step) : 149.28M (pg-node201, per node)
Full Max Mem usage  : 149.28M
Total Disk Read     : 262.98M
Total Disk Write    : 6.46M


Acknowledgements:
=================
"""

def extract_log(data: str):
    num_results = data.count("Job ID")
    df = pd.DataFrame(columns=['job_name', 'job_state', 'job_submit', 'job_start', 'job_end', 'job_walltime', 'job_mem', 'job_full_mem'], index=range(num_results))
    lines = data.split("\n")
    i = 0
    for line in lines:
        if "Name" in line:
            df.loc[i, 'job_name'] = line.split(":")[1].strip()
        if "State" in line:
            df.loc[i, 'job_state'] = line.split(":")[1].strip()
        if "Submit" in line:
            df.loc[i, 'job_submit'] = ":".join(line.split(":")[1:]).strip()
        if "Start" in line:
            df.loc[i, 'job_start'] = ":".join(line.split(":")[1:]).strip()
        if "End" in line:
            df.loc[i, 'job_end'] = ":".join(line.split(":")[1:]).strip()
        if "Used walltime" in line:
            df.loc[i, 'job_walltime'] = ":".join(line.split(":")[1:]).strip()
        if "Mem reserved" in line:
            df.loc[i, 'job_mem'] = line.split(":")[1].strip()
        if "Full Max Mem usage" in line:
            df.loc[i, 'job_full_mem'] = line.split(":")[1].strip()
            i += 1

    return df

def load_data(path: Path = Path("results/data/aggregate_log.txt")) -> str:
    with open(path, 'r', encoding="ISO-8859-1") as file:
        data = file.read()
    return data

def main():
    data = load_data()
    df = extract_log(data)
    out_path = Path("results/data/metrics.xlsx")
    print(df)
    with pd.ExcelWriter(out_path) as writer:
        df.to_excel(writer)

if __name__ == "__main__":
    main()