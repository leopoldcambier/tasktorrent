FROM ubuntu:bionic

RUN apt-get update
RUN apt-get install -y wget gcc g++ python3.7 make ssh
RUN apt-get install -y openmpi-bin libopenmpi-dev
RUN apt-get install -y libscotch-dev
RUN apt-get install -y libopenblas-dev liblapacke-dev
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1-Linux-x86_64.tar.gz -O cmake-3.19.1.tar.gz && mkdir cmake && tar -xvf cmake-3.19.1.tar.gz -C cmake --strip-components 1

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2 -O eigen-3.3.7.tar.bz2 && mkdir eigen && tar -xvf eigen-3.3.7.tar.bz2 -C eigen --strip-components 1

RUN wget https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2020.10.0.tar.gz -O upcxx-2020.10.0.tar.gz && mkdir upcxx && tar -xvzf upcxx-2020.10.0.tar.gz -C upcxx --strip-component 1
RUN cd upcxx && ./configure && make all && make install

RUN apt-get install -y git

ENV PATH "$PATH:/usr/local/upcxx/bin"
ENV PATH "$PATH:/cmake/bin"
