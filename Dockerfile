FROM ubuntu:latest
MAINTAINER your_name "email@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN pip install -r requirements.txt
