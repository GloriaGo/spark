#!/bin/bash

./build/mvn -pl mllib clean package -Dmaven.test.skip=true -T 4

##local test
scp mllib/target/spark-mllib_2.11-2.3.0-SNAPSHOT.jar ~/Downloads/spark-2.1.0-bin-hadoop2.6/jars/

##clusters test
#scp mllib/target/spark-mllib_2.11-2.3.0-SNAPSHOT.jar guoyu@10.111.1.26:/mnt/local/guoyu/
#ssh guoyu@bach16 "mv /mnt/local/guoyu/spark-mllib_2.11-2.3.0-SNAPSHOT.jar /mnt/local/guoyu/spark-2.1.1-bin-hadoop2.7/jars/"

