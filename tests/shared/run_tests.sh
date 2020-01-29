#!/bin/bash

./tests_serialize

if [ $? != "0" ]
then
        exit 1
fi

./tests 32 1 32 0
