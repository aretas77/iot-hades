FROM python:3

RUN apt-get update && apt-get install bash
RUN pip3 install tensorflow numpy tf-agents paho-mqtt

WORKDIR /app
COPY . /app

#ENTRYPOINT [ "/bin/bash", "./start.sh" ]
