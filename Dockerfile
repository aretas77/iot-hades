FROM python:3

RUN apt-get update

WORKDIR /app

RUN pip3 install tensorflow numpy tf-agents

COPY . /app

ENTRYPOINT [ "/bin/bash", "./start.sh" ]
