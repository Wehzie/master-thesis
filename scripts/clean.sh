rm *.log
rm parallel_job_launcher.py
rm job.sh
echo "also clean data? [y/n]"
read answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
	echo "Deleting data/*"
	rm -r data/*
else
	echo "Not deleting data/*"
fi