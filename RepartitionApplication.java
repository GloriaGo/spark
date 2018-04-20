package mllibExample;

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// $example on$
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.DataFrameWriter;
// $example off$

public class JavaLDAExample {

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("Repartition")
                .getOrCreate();
        Dataset<Row> dataset = spark.read().format("libsvm")
                .load(args[0]);
        DataFrameWriter<Row> data = dataset.repartition(8).cache().write().format("libsvm");
        data.save(args[1]);
        spark.stop();
    }
}
