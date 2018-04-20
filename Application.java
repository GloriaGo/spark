package mllibExample;

import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class TestBatchsize{
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("OnlineLDA")
                .getOrCreate();
        Dataset<Row> dataset = spark.read().format("libsvm")
                .load(args[0]);
        LDA lda = new LDA().setOptimizer("online")
                .setSubsamplingRate(Double.parseDouble(args[1]))   // batchsize
                .setLearningDecay(Double.parseDouble(args[2]))   // kappa
                .setLearningOffset(Integer.parseInt(args[3]))   // tau0
                .setMaxIter(Integer.parseInt(args[4]))     // iteration
                .setK(Integer.parseInt(args[5]))   // topic number K
                .setSeed(Long.parseLong(args[6]));  // validate data generate based on this seed

        LDAModel model = lda.fit(dataset);

        Dataset<Row> testset = spark.read().format("libsvm")
                .load(args[7]).repartition(Integer.parseInt(args[8])).cache();
        double lp = model.logPerplexity(testset);
        System.out.println("YY=Test data result=The upper bound on logperplexity: " + lp);
    }
}
