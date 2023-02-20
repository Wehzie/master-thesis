# aggregate results and log files after all jobs are done

# aggregate log files
tail -n 45 job-*.log > aggregate_log.txt
rm job*.log
mv aggregate_log.txt data/

# aggregate results
cd data
mkdir -p quantitative_experiment_hybrid
mkdir -p quantitative_experiment_python
mv quantitative_experiment_hybrid*/* quantitative_experiment_hybrid/
mv quantitative_experiment_python*/* quantitative_experiment_python/
rm -r quantitative_experiment_hybrid_sweep*
rm -r quantitative_experiment_python_sweep*

# aggregate qualitative
mkdir -p data/agg_qualitative_experiments
mv data/qualitative_* data/agg_qualitative_experiments/

# extract runtime and memory usage from log files
python3 scripts/extract_log.py