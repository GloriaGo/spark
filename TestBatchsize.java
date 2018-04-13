package mllibExample;

import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
 import java.io.*;

public class TestBatchsize{

    public static void main(String[] args) {
        // Local test
//        SparkSession spark = SparkSession
//                .builder()
//                .appName("TestBatchsize").master("local[2]")
//                .getOrCreate();
//        Dataset<Row> dataset = spark.read().format("libsvm")
//                .load("datasets/small4Samples.txt").repartition(2).cache();
//        LDA lda = new LDA().setOptimizer("online")
//                .setSubsamplingRate(1)   // batchsize
//                .setLearningDecay(0.7)   // kappa
//                .setLearningOffset(1024)   // tau0
//                .setMaxIter(1)     // iteration
//                .setK(2)   // topic number K
//                .setSeed(10l);
//        System.out.println("getSeed:"+lda.getSeed());
//        LDAModel model = lda.fit(dataset);
//        double ll = model.logLikelihood(dataset);
//        double lp = model.logPerplexity(dataset);
//        System.out.println("The lower bound on the log likelihood of the entire corpus: " + ll);
//        System.out.println("The upper bound on perplexity: " + lp);

        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("LearningRate")
                .getOrCreate();
        Dataset<Row> dataset = spark.read().format("libsvm")
                .load(args[0]);

                    LDA lda = new LDA().setOptimizer("online")
                            .setSubsamplingRate(Double.parseDouble(args[1]))   // batchsize
                            .setLearningDecay(Double.parseDouble(args[2]))   // kappa
                            .setLearningOffset(Integer.parseInt(args[3]))   // tau0
                            .setMaxIter(Integer.parseInt(args[4]))     // iteration
                            .setK(Integer.parseInt(args[5]))   // topic number K
                            .setSeed(Long.parseLong(args[8]));

        LDAModel model = lda.fit(dataset);

        Dataset<Row> testset =spark.read().format("libsvm")
                .load(args[7]).repartition(Integer.parseInt(args[6])).cache();

//        double ll = model.logLikelihood(testset);
        double lp = model.logPerplexity(testset);

//        System.out.println("YY=The lower bound on the log likelihood of the entire corpus: " + ll);
        System.out.println("YY=The upper bound on logperplexity: " + lp);

        // Describe topics.
//        Dataset<Row> topics = model.describeTopics(20);
//        System.out.println("The topics described by their top-weighted terms:");
//        topics.show(20,false);
//        String output = args[8];
//        try {
//            BufferedWriter writer = new BufferedWriter(new FileWriter(output));
//            for(Row x : topics.collectAsList()) {
//                writer.write(x.toString());
//                writer.newLine();
//            }
//            writer.close();
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }
}
