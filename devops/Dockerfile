FROM ubuntu:rolling

WORKDIR /usr/src/app

# copy from source (local dir) to dest (docker filesystem)
COPY . .

RUN apt-get update && apt-get install python3 python3-pip python-is-python3 ngspice -y
RUN pip3 install -r requirements.txt

# run experiment
#   docker exec -it <container> python3 src/experiment.py 