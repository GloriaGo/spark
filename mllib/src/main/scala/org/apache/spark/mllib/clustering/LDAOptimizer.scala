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

package org.apache.spark.mllib.clustering

import java.util.Random

import breeze.linalg.{all, normalize, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.{abs, digamma, exp, trigamma}
import breeze.stats.distributions.{Gamma, RandBasis}
import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.TaskContext
import org.apache.spark.graphx._
import org.apache.spark.graphx.util.PeriodicGraphCheckpointer
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD

@Since("1.4.0")
@DeveloperApi
trait LDAOptimizer {

  /*
    DEVELOPERS NOTE:
    An LDAOptimizer contains an algorithm for LDA and performs the actual computation, which
    stores internal data structure (Graph or Matrix) and other parameters for the algorithm.
    The interface is isolated to improve the extensibility of LDA.
   */

  /**
    * Initializer for the optimizer. LDA passes the common parameters to the optimizer and
    * the internal structure can be initialized properly.
    */
  private[clustering] def initialize(docs: RDD[(Long, Vector)], lda: LDA): LDAOptimizer

  private[clustering] def next(): LDAOptimizer

  private[clustering] def getLDAModel(iterationTimes: Array[Double]): LDAModel
}

@Since("1.4.0")
@DeveloperApi
final class EMLDAOptimizer extends LDAOptimizer {

  import LDA._

  // Adjustable parameters
  private var keepLastCheckpoint: Boolean = true

  /**
    * If using checkpointing, this indicates whether to keep the last checkpoint (vs clean up).
    */
  @Since("2.0.0")
  def getKeepLastCheckpoint: Boolean = this.keepLastCheckpoint

  /**
    * If using checkpointing, this indicates whether to keep the last checkpoint (vs clean up).
    * Deleting the checkpoint can cause failures if a data partition is lost, so set this bit with
    * care.
    *
    * Default: true
    *
    * @note Checkpoints will be cleaned up via reference counting, regardless.
    */
  @Since("2.0.0")
  def setKeepLastCheckpoint(keepLastCheckpoint: Boolean): this.type = {
    this.keepLastCheckpoint = keepLastCheckpoint
    this
  }

  // The following fields will only be initialized through the initialize() method
  private[clustering] var graph: Graph[TopicCounts, TokenCount] = null
  private[clustering] var k: Int = 0
  private[clustering] var vocabSize: Int = 0
  private[clustering] var docConcentration: Double = 0
  private[clustering] var topicConcentration: Double = 0
  private[clustering] var checkpointInterval: Int = 10
  private var graphCheckpointer: PeriodicGraphCheckpointer[TopicCounts, TokenCount] = null

  /**
    * Compute bipartite term/doc graph.
    */
  override private[clustering] def initialize(
                                               docs: RDD[(Long, Vector)],
                                               lda: LDA): EMLDAOptimizer = {
    // EMLDAOptimizer currently only supports symmetric document-topic priors
    val docConcentration = lda.getDocConcentration

    val topicConcentration = lda.getTopicConcentration
    val k = lda.getK

    // Note: The restriction > 1.0 may be relaxed in the future (allowing sparse solutions),
    // but values in (0,1) are not yet supported.
    require(docConcentration > 1.0 || docConcentration == -1.0, s"LDA docConcentration must be" +
      s" > 1.0 (or -1 for auto) for EM Optimizer, but was set to $docConcentration")
    require(topicConcentration > 1.0 || topicConcentration == -1.0, s"LDA topicConcentration " +
      s"must be > 1.0 (or -1 for auto) for EM Optimizer, but was set to $topicConcentration")

    this.docConcentration = if (docConcentration == -1) (50.0 / k) + 1.0 else docConcentration
    this.topicConcentration = if (topicConcentration == -1) 1.1 else topicConcentration
    val randomSeed = lda.getSeed

    // For each document, create an edge (Document -> Term) for each unique term in the document.
    val edges: RDD[Edge[TokenCount]] = docs.flatMap { case (docID: Long, termCounts: Vector) =>
      // Add edges for terms with non-zero counts.
      termCounts.asBreeze.activeIterator.filter(_._2 != 0.0).map { case (term, cnt) =>
        Edge(docID, term2index(term), cnt)
      }
    }

    // Create vertices.
    // Initially, we use random soft assignments of tokens to topics (random gamma).
    val docTermVertices: RDD[(VertexId, TopicCounts)] = {
      val verticesTMP: RDD[(VertexId, TopicCounts)] =
        edges.mapPartitionsWithIndex { case (partIndex, partEdges) =>
          val random = new Random(partIndex + randomSeed)
          partEdges.flatMap { edge =>
            val gamma = normalize(BDV.fill[Double](k)(random.nextDouble()), 1.0)
            val sum = gamma * edge.attr
            Seq((edge.srcId, sum), (edge.dstId, sum))
          }
        }
      verticesTMP.reduceByKey(_ + _)
    }

    // Partition such that edges are grouped by document
    this.graph = Graph(docTermVertices, edges).partitionBy(PartitionStrategy.EdgePartition1D)
    this.k = k
    this.vocabSize = docs.take(1).head._2.size
    this.checkpointInterval = lda.getCheckpointInterval
    this.graphCheckpointer = new PeriodicGraphCheckpointer[TopicCounts, TokenCount](
      checkpointInterval, graph.vertices.sparkContext)
    this.graphCheckpointer.update(this.graph)
    this.globalTopicTotals = computeGlobalTopicTotals()
    this
  }

  override private[clustering] def next(): EMLDAOptimizer = {
    require(graph != null, "graph is null, EMLDAOptimizer not initialized.")

    val eta = topicConcentration
    val W = vocabSize
    val alpha = docConcentration

    val N_k = globalTopicTotals
    val sendMsg: EdgeContext[TopicCounts, TokenCount, (Boolean, TopicCounts)] => Unit =
      (edgeContext) => {
        // Compute N_{wj} gamma_{wjk}
        val N_wj = edgeContext.attr
        // E-STEP: Compute gamma_{wjk} (smoothed topic distributions), scaled by token count
        // N_{wj}.
        val scaledTopicDistribution: TopicCounts =
        computePTopic(edgeContext.srcAttr, edgeContext.dstAttr, N_k, W, eta, alpha) *= N_wj
        edgeContext.sendToDst((false, scaledTopicDistribution))
        edgeContext.sendToSrc((false, scaledTopicDistribution))
      }
    // The Boolean is a hack to detect whether we could modify the values in-place.
    // TODO: Add zero/seqOp/combOp option to aggregateMessages. (SPARK-5438)
    val mergeMsg: ((Boolean, TopicCounts), (Boolean, TopicCounts)) => (Boolean, TopicCounts) =
    (m0, m1) => {
      val sum =
        if (m0._1) {
          m0._2 += m1._2
        } else if (m1._1) {
          m1._2 += m0._2
        } else {
          m0._2 + m1._2
        }
      (true, sum)
    }
    // M-STEP: Aggregation computes new N_{kj}, N_{wk} counts.
    val docTopicDistributions: VertexRDD[TopicCounts] =
      graph.aggregateMessages[(Boolean, TopicCounts)](sendMsg, mergeMsg)
        .mapValues(_._2)
    // Update the vertex descriptors with the new counts.
    val newGraph = Graph(docTopicDistributions, graph.edges)
    graph = newGraph
    graphCheckpointer.update(newGraph)
    globalTopicTotals = computeGlobalTopicTotals()
    this
  }

  /**
    * Aggregate distributions over topics from all term vertices.
    *
    * Note: This executes an action on the graph RDDs.
    */
  private[clustering] var globalTopicTotals: TopicCounts = null

  private def computeGlobalTopicTotals(): TopicCounts = {
    val numTopics = k
    graph.vertices.filter(isTermVertex).values.fold(BDV.zeros[Double](numTopics))(_ += _)
  }

  override private[clustering] def getLDAModel(iterationTimes: Array[Double]): LDAModel = {
    require(graph != null, "graph is null, EMLDAOptimizer not initialized.")
    val checkpointFiles: Array[String] = if (keepLastCheckpoint) {
      this.graphCheckpointer.deleteAllCheckpointsButLast()
      this.graphCheckpointer.getAllCheckpointFiles
    } else {
      this.graphCheckpointer.deleteAllCheckpoints()
      Array.empty[String]
    }
    // The constructor's default arguments assume gammaShape = 100 to ensure equivalence in
    // LDAModel.toLocal conversion.
    new DistributedLDAModel(this.graph, this.globalTopicTotals, this.k, this.vocabSize,
      Vectors.dense(Array.fill(this.k)(this.docConcentration)), this.topicConcentration,
      iterationTimes, DistributedLDAModel.defaultGammaShape, checkpointFiles)
  }
}


@Since("1.4.0")
@DeveloperApi
final class OnlineLDAOptimizer extends LDAOptimizer with Logging {

  // LDA common parameters
  private var k: Int = 0
  private var corpusSize: Long = 0
  private var vocabSize: Int = 0

  /** alias for docConcentration */
  private var alpha: Vector = Vectors.dense(0)

  /** (for debugging)  Get docConcentration */
  private[clustering] def getAlpha: Vector = alpha

  /** alias for topicConcentration */
  private var eta: Double = 0

  /** (for debugging)  Get topicConcentration */
  private[clustering] def getEta: Double = eta

  private var randomGenerator: java.util.Random = null

  /** (for debugging) Whether to sample mini-batches with replacement. (default = true) */
  private var sampleWithReplacement: Boolean = true

  // Online LDA specific parameters
  // Learning rate is: (tau0 + t)^{-kappa}
  private var tau0: Double = 1024
  private var kappa: Double = 0.51
  private var miniBatchFraction: Double = 0.05
  private var optimizeDocConcentration: Boolean = false

  // internal data structure
  private var docs: RDD[(Long, Vector)] = null

  /** Dirichlet parameter for the posterior over topics */
  private var lambda: BDM[Double] = null

  /** (for debugging) Get parameter for topics */
  private[clustering] def getLambda: BDM[Double] = lambda

  /** Current iteration (count of invocations of [[next()]]) */
  private var iteration: Int = 0
  private var gammaShape: Double = 100

  /**
    * A (positive) learning parameter that downweights early iterations. Larger values make early
    * iterations count less.
    */
  @Since("1.4.0")
  def getTau0: Double = this.tau0

  /**
    * A (positive) learning parameter that downweights early iterations. Larger values make early
    * iterations count less.
    * Default: 1024, following the original Online LDA paper.
    */
  @Since("1.4.0")
  def setTau0(tau0: Double): this.type = {
    require(tau0 > 0, s"LDA tau0 must be positive, but was set to $tau0")
    this.tau0 = tau0
    this
  }

  /**
    * Learning rate: exponential decay rate
    */
  @Since("1.4.0")
  def getKappa: Double = this.kappa

  /**
    * Learning rate: exponential decay rate---should be between
    * (0.5, 1.0] to guarantee asymptotic convergence.
    * Default: 0.51, based on the original Online LDA paper.
    */
  @Since("1.4.0")
  def setKappa(kappa: Double): this.type = {
    require(kappa >= 0, s"Online LDA kappa must be nonnegative, but was set to $kappa")
    this.kappa = kappa
    this
  }

  /**
    * Mini-batch fraction, which sets the fraction of document sampled and used in each iteration
    */
  @Since("1.4.0")
  def getMiniBatchFraction: Double = this.miniBatchFraction

  /**
    * Mini-batch fraction in (0, 1], which sets the fraction of document sampled and used in
    * each iteration.
    *
    * @note This should be adjusted in synch with `LDA.setMaxIterations()`
    * so the entire corpus is used.  Specifically, set both so that
    * maxIterations * miniBatchFraction is at least 1.
    *
    * Default: 0.05, i.e., 5% of total documents.
    */
  @Since("1.4.0")
  def setMiniBatchFraction(miniBatchFraction: Double): this.type = {
    require(miniBatchFraction > 0.0 && miniBatchFraction <= 1.0,
      s"Online LDA miniBatchFraction must be in range (0,1], but was set to $miniBatchFraction")
    this.miniBatchFraction = miniBatchFraction
    this
  }
  /**
    * Optimize docConcentration, indicates whether docConcentration (Dirichlet parameter for
    * document-topic distribution) will be optimized during training.
    */
  @Since("1.5.0")
  def getOptimizeDocConcentration: Boolean = this.optimizeDocConcentration

  /**
    * Sets whether to optimize docConcentration parameter during training.
    *
    * Default: false
    */
  @Since("1.5.0")
  def setOptimizeDocConcentration(optimizeDocConcentration: Boolean): this.type = {
    this.optimizeDocConcentration = optimizeDocConcentration
    this
  }

  /**
    * Set the Dirichlet parameter for the posterior over topics.
    * This is only used for testing now. In the future, it can help support training stop/resume.
    */
  private[clustering] def setLambda(lambda: BDM[Double]): this.type = {
    this.lambda = lambda
    this
  }

  /**
    * Used for random initialization of the variational parameters.
    * Larger value produces values closer to 1.0.
    * This is only used for testing currently.
    */
  private[clustering] def setGammaShape(shape: Double): this.type = {
    this.gammaShape = shape
    this
  }

  /**
    * Sets whether to sample mini-batches with or without replacement. (default = true)
    * This is only used for testing currently.
    */
  private[clustering] def setSampleWithReplacement(replace: Boolean): this.type = {
    this.sampleWithReplacement = replace
    this
  }

  override private[clustering] def initialize(
                                               docs: RDD[(Long, Vector)],
                                               lda: LDA): OnlineLDAOptimizer = {
    this.k = lda.getK
    this.corpusSize = docs.count()
    this.vocabSize = docs.first()._2.size
    this.alpha = if (lda.getAsymmetricDocConcentration.size == 1) {
      if (lda.getAsymmetricDocConcentration(0) == -1) Vectors.dense(Array.fill(k)(1.0 / k))
      else {
        require(lda.getAsymmetricDocConcentration(0) >= 0,
          s"all entries in alpha must be >=0, got: $alpha")
        Vectors.dense(Array.fill(k)(lda.getAsymmetricDocConcentration(0)))
      }
    } else {
      require(lda.getAsymmetricDocConcentration.size == k,
        s"alpha must have length k, got: $alpha")
      lda.getAsymmetricDocConcentration.foreachActive { case (_, x) =>
        require(x >= 0, s"all entries in alpha must be >= 0, got: $alpha")
      }
      lda.getAsymmetricDocConcentration
    }
    this.eta = if (lda.getTopicConcentration == -1) 1.0 / k else lda.getTopicConcentration
    this.randomGenerator = new Random(lda.getSeed)

    this.docs = docs

    logInfo(s"YY=topic:${k}=miniBatchFraction:${miniBatchFraction}=tau0:${tau0}=kappa:${kappa}" +
      s"=ModelAveraging=corpusSize:${corpusSize}=randomSeed=${lda.getSeed}")

    // Initialize the variational distribution q(beta|lambda)
    this.lambda = getGammaMatrix(k, vocabSize)
    this.iteration = 0
    this
  }


  override private[clustering] def next(): OnlineLDAOptimizer = {
    val batch = docs.sample(withReplacement = sampleWithReplacement, miniBatchFraction,
      randomGenerator.nextLong())
    // YY ignore
    // if (batch.isEmpty()) return this
    submitMiniBatch(batch)
  }

  /**
   * Submit a subset (like 1%, decide by the miniBatchFraction) of the corpus to the Online LDA
   * model, and it will update the topic distribution adaptively for the terms appearing in the
   * subset.
   */
  private[clustering] def submitMiniBatch(batch: RDD[(Long, Vector)]): OnlineLDAOptimizer = {
    iteration += 1
    val iter = this.iteration
    val k = this.k
    val vocabSize = this.vocabSize
    // Original DirichletExp and BroadCast
//    val expElogbeta = exp(LDAUtils.dirichletExpectation(lambda)).t
//    val expElogbetaBc = batch.sparkContext.broadcast(expElogbeta)

    // YY Model Averaging Only BroadCast, no beta calculation 0.18s
    val lambdaBc = batch.sparkContext.broadcast(lambda)  //  v * k, 1.5 s
    // YY need params (1e5)
    val weight = rho()
    val A1 = 1.0 - weight
    val A2 = weight * this.corpusSize
    val A3 = weight * this.eta
    val workerSize = 8.0
    // TODO: executor size should be read from setting or environment

    val alpha = this.alpha.asBreeze
    val gammaShape = this.gammaShape
    val optimizeDocConcentration = this.optimizeDocConcentration
    // If and only if optimizeDocConcentration is set true,
    // we calculate logphat in the same pass as other statistics.
    // No calculation of loghat happens otherwise.
    val logphatPartOptionBase = () => if (optimizeDocConcentration) {
      Some(BDV.zeros[Double](k))
    } else {
      None
    }

    val stats: RDD[(BDM[Double], Option[BDV[Double]], Long)] = batch.mapPartitions { docs =>
      val nonEmptyDocs = docs.filter(_._2.numNonzeros > 0)

      // YY Sparse-Update Initialize
      var multiA1 = 1.0
      var sumA1 = 0.0
      val QLambda : BDM[Double] = lambdaBc.value  // v * k, 2.7s
      var QSum : BDV[Double] = sum(QLambda(breeze.linalg.*, ::)) // k, 0.5s
      var QLambdaD = QLambda.t  // 2000 nano for whole matrix

      val stat = BDM.zeros[Double](k, vocabSize)    // k * v, 0.02 s
      val logphatPartOption = logphatPartOptionBase()
      var nonEmptyDocCount: Long = 0L
      nonEmptyDocs.foreach { case (_, termCounts: Vector) =>
        nonEmptyDocCount += 1
        // YY Sparse Words, 255 nano per id
        val (ids: List[Int], cts: Array[Double]) = termCounts match {
          case v: DenseVector => ((0 until v.size).toList, v.values)
          case v: SparseVector => (v.indices.toList, v.values)
        }

        // YY Sparse Calculate expElogbeta  (50%)
        val partQD = QLambdaD(ids, ::).toDenseMatrix  //  ids * k, 62 nano per element
        val tmp1 = A3 * sumA1
        val tmp2 = tmp1 * vocabSize

        val QAlphaD = partQD * multiA1 + tmp1 //  ids * k, 8 nano per element
        val rowSum = QSum * multiA1 + tmp2  // k, 9 nano per element

        val digAlphaD = digamma(QAlphaD)  //  ids * k, 670 nano per element
        val digRowSum = digamma(rowSum) //  k, 90 nano per element
        val result = digAlphaD(breeze.linalg.*, ::) - digRowSum //  ids * k, 14 nano per element

        val newPartExpElogBeta = exp(result).toDenseMatrix  // ids * k, 95 nano per element

        // YY Local VI with sparse expElogbeta  (40%)
        // sstats : k * ids, 9 ms, 195 nano per element
        val (gammad, sstats) = OnlineLDAOptimizer.newVariationalTopicInference(
          termCounts, newPartExpElogBeta, alpha, gammaShape, k, iter, ids, cts)

        // YY real delta and sparse delta Sparse-Update (5%)
        val delta : BDM[Double] = sstats *:* newPartExpElogBeta.t // k * ids, 5.5 nano per element
        val tmp3 = A2 / (multiA1 * A1)
        val newDelta = (delta * tmp3).toDenseMatrix   // k * ids, 7 nano per element
        QLambdaD(ids, ::) := partQD + newDelta.t  // 45 nano per element
        val deltaSum = sum(newDelta(breeze.linalg.*, ::))   // 3 nano per element
        QSum := QSum + deltaSum
        sumA1 = sumA1 + multiA1
        multiA1 = multiA1 * A1

        // Original
        // stat(::, ids) := stat(::, ids) + sstats
        // logphatPartOption.foreach(_ += LDAUtils.dirichletExpectation(gammad))
      }
      // YY Model Averaging
      val tmp4 = A3 * sumA1
      stat := QLambdaD.t * multiA1 + tmp4// 0.27 s for whole Matrix
      Iterator((stat, logphatPartOption, nonEmptyDocCount))
    }

    val elementWiseSum = (
                           u: (BDM[Double], Option[BDV[Double]], Long),
                           v: (BDM[Double], Option[BDV[Double]], Long)) => {
      u._1 += v._1
      u._2.foreach(_ += v._2.get)
      (u._1, u._2, u._3 + v._3)
    }

    val (statsSum: BDM[Double], logphatOption: Option[BDV[Double]], nonEmptyDocsN: Long) = stats
      .treeAggregate((BDM.zeros[Double](k, vocabSize), logphatPartOptionBase(), 0L))(
        elementWiseSum, elementWiseSum
      )

    // Original broadcast value
    // expElogbetaBc.destroy(false)
    // YY broadcast value
    lambdaBc.destroy(blocking = false)

    if (nonEmptyDocsN == 0) {
      logWarning("No non-empty documents were submitted in the batch.")
      // Therefore, there is no need to update any of the model parameters
      return this
    }
    // Original update gloal parameter lambda
//    val batchResult = statsSum *:* expElogbeta.t  // 0.2s
//    // Note that this is an optimization to avoid batch.count
//    val batchSize = (miniBatchFraction * corpusSize).ceil.toInt
//    updateLambda(batchResult, batchSize)   // 1s

    // YY Model Averaging the global parameter
    val newLambda : BDM[Double] = statsSum /:/ workerSize
    setLambda(newLambda)  // 0.13 s for whole matrix
    // YY ignore
    //    logphatOption.foreach(_ /= nonEmptyDocsN.toDouble)
    //    logphatOption.foreach(updateAlpha(_, nonEmptyDocsN))

    this
  }

  /**
    * Update lambda based on the batch submitted. batchSize can be different for each iteration.
    */
  private def updateLambda(stat: BDM[Double], batchSize: Int): Unit = {
    // weight of the mini-batch.
    val weight = rho()

    // Update lambda based on documents.
    lambda := (1 - weight) * lambda +
      weight * (stat * (corpusSize.toDouble / batchSize.toDouble) + eta)
  }

  /**
    * Update alpha based on `logphat`.
    * Uses Newton-Rhapson method.
    * @see Section 3.3, Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters
    *      (http://jonathan-huang.org/research/dirichlet/dirichlet.pdf)
    * @param logphat Expectation of estimated log-posterior distribution of
    *                topics in a document averaged over the batch.
    * @param nonEmptyDocsN number of non-empty documents
    */
  private def updateAlpha(logphat: BDV[Double], nonEmptyDocsN: Double): Unit = {
    val weight = rho()
    val alpha = this.alpha.asBreeze.toDenseVector

    val gradf = nonEmptyDocsN * (-LDAUtils.dirichletExpectation(alpha) + logphat)

    val c = nonEmptyDocsN * trigamma(sum(alpha))
    val q = -nonEmptyDocsN * trigamma(alpha)
    val b = sum(gradf / q) / (1D / c + sum(1D / q))

    val dalpha = -(gradf - b) / q

    if (all((weight * dalpha + alpha) >:> 0D)) {
      alpha :+= weight * dalpha
      this.alpha = Vectors.dense(alpha.toArray)
    }
  }

  /** Calculate learning rate rho for the current [[iteration]]. */
  private def rho(): Double = {
    math.pow(getTau0 + this.iteration, -getKappa)
  }

  /**
    * Get a random matrix to initialize lambda.
    */
  private def getGammaMatrix(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](col, row, temp).t
  }

  override private[clustering] def getLDAModel(iterationTimes: Array[Double]): LDAModel = {
    new LocalLDAModel(Matrices.fromBreeze(lambda).transpose, alpha, eta, gammaShape)
  }

}
/**
 * Serializable companion object containing helper methods and shared code for
 * [[OnlineLDAOptimizer]] and [[LocalLDAModel]].
 */
private[clustering] object OnlineLDAOptimizer extends Logging{
  /**
    * Uses variational inference to infer the topic distribution `gammad` given the term counts
    * for a document. `termCounts` must contain at least one non-zero entry, otherwise Breeze will
    * throw a BLAS error.
    *
    * An optimization (Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001)
    * avoids explicit computation of variational parameter `phi`.
    * @see <a href="http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.31.7566">here</a>
    *
    * @return Returns a tuple of `gammad` - estimate of gamma, the topic distribution, `sstatsd` -
    *         statistics for updating lambda and `ids` - list of termCounts vector indices.
    */
  private[clustering] def variationalTopicInference(
                                                     termCounts: Vector,
                                                     expElogbeta: BDM[Double],
                                                     alpha: breeze.linalg.Vector[Double],
                                                     gammaShape: Double,
                                                     k: Int): (BDV[Double], BDM[Double], List[Int]) = {
    val (ids: List[Int], cts: Array[Double]) = termCounts match {
      case v: DenseVector => ((0 until v.size).toList, v.values)
      case v: SparseVector => (v.indices.toList, v.values)
    }
    // Initialize the variational distribution q(theta|gamma) for the mini-batch
    val gammad: BDV[Double] =
      new Gamma(gammaShape, 1.0 / gammaShape).samplesVector(k)                   // K
    val expElogthetad: BDV[Double] = exp(LDAUtils.dirichletExpectation(gammad))  // K
    val expElogbetad = expElogbeta(ids, ::).toDenseMatrix                        // ids * K

    val phiNorm: BDV[Double] = expElogbetad * expElogthetad +:+ 1e-100            // ids
    var meanGammaChange = 1D
    val ctsVector = new BDV[Double](cts)                                         // ids

    // Iterate between gamma and phi until convergence
    while (meanGammaChange > 1e-3) {
      val lastgamma = gammad.copy
      //        K                  K * ids               ids
      gammad := (expElogthetad *:* (expElogbetad.t * (ctsVector /:/ phiNorm))) +:+ alpha
      expElogthetad := exp(LDAUtils.dirichletExpectation(gammad))
      // TODO: Keep more values in log space, and only exponentiate when needed.
      phiNorm := expElogbetad * expElogthetad +:+ 1e-100
      meanGammaChange = sum(abs(gammad - lastgamma)) / k
    }

    val sstatsd = expElogthetad.asDenseMatrix.t * (ctsVector /:/ phiNorm).asDenseMatrix
    (gammad, sstatsd, ids)
  }

  /**
    * YY motified a param to fit Model Average.
    * @param expElogbetad is the same as expElogbetad in original function
    * @return Returns a tuple of `gammad` - estimate of gamma, the topic distribution, `sstatsd` -
    *         statistics for updating lambda and `ids` - list of termCounts vector indices.
    */
  private[clustering] def newVariationalTopicInference(
                                                     termCounts: Vector,
                                                     expElogbetad: BDM[Double],
                                                     alpha: breeze.linalg.Vector[Double],
                                                     gammaShape: Double,
                                                     k: Int,
                                                     iter: Int,
                                                     ids: List[Int],
                                                     cts: Array[Double]): (BDV[Double], BDM[Double]) = {
    // Initialize the variational distribution q(theta|gamma) for the mini-batch
    val gammad: BDV[Double] =
      new Gamma(gammaShape, 1.0 / gammaShape).samplesVector(k)                   // K
    val expElogthetad: BDV[Double] = exp(LDAUtils.dirichletExpectation(gammad))  // K
    // YY ignored the original version
    // val expElogbetad = expElogbeta(ids, ::).toDenseMatrix                        // ids * K

    val phiNorm: BDV[Double] = expElogbetad * expElogthetad +:+ 1e-100            // ids
    var meanGammaChange = 1D
    val ctsVector = new BDV[Double](cts)                                         // ids

    // Iterate between gamma and phi until convergence
    while (meanGammaChange > 1e-3) {
      val lastgamma = gammad.copy
      //        K                  K * ids               ids
      gammad := (expElogthetad *:* (expElogbetad.t * (ctsVector /:/ phiNorm))) +:+ alpha
      expElogthetad := exp(LDAUtils.dirichletExpectation(gammad))
      // TODO: Keep more values in log space, and only exponentiate when needed.
      phiNorm := expElogbetad * expElogthetad +:+ 1e-100
      meanGammaChange = sum(abs(gammad - lastgamma)) / k
    }

    val sstatsd = expElogthetad.asDenseMatrix.t * (ctsVector /:/ phiNorm).asDenseMatrix
    (gammad, sstatsd)
  }

  private[clustering] def YYLog(
                                 keyWord: String,
                                 duration: Double,
                                 iter: Int
                               ): Unit = {
    logInfo(s"YYY=Iteration:${iter}=PartitionID:${TaskContext.getPartitionId()}" +
      s"=${keyWord}:${duration}")
  }
}
