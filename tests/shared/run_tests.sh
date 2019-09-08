#!/bin/bash

(make clean && make all) || exit 1

./tests_serialize

if [ $? != "0" ]
then
        exit 1
fi

./tests 32 1 32 0
