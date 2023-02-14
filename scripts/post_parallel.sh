# aggregate results and log files after all jobs are done

# aggregate log files
tail -n 50 *.log > aggregate_log.txt
rm job*.log
mv aggregate_log.txt data/

# aggregate results
mkdir -p data/quantitative_experiment_hybrid
mkdir -p data/quantitative_experiment_python
mv data/quantitative_experiment_hybrid*/* data/quantitative_experiment_hybrid/
mv data/quantitative_experiment_python*/* data/quantitative_experiment_python/
rm -r data/quantitative_experiment_hybrid_sweep*
rm -r data/quantitative_experiment_python_sweep*

# extract runtime and memory usage from log files
python3 scripts/extract_log.py