#!/bin/bash

./tests_serialize

if [ $? != "0" ]
then
        exit 1
fi

./tests 8 0 0 0 --gtest_filter=graph.mini --gtest_repeat=100

if [ $? != "0" ]
then
        exit 1
fi

./tests 8 0 0 0 --gtest_filter=reduction.* --gtest_repeat=100

if [ $? != "0" ]
then
        exit 1
fi

./tests 32 1 32 0

if [ $? != "0" ]
then
        exit 1
fi