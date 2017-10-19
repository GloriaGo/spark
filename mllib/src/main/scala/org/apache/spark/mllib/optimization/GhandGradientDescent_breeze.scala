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

package org.apache.spark.mllib.optimization

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector => BDV, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.TaskContext


/**
  * This class use the Breeze library to see whether the linalg used in treeaggregate() is the bottleneck.
  */
/**
 * Class used to solve an optimization problem using Gradient Descent.
  *
  * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */


class GhandGradientDescent_breeze private[spark] (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
//  private var convergenceTol: Double = 0.001
  private var convergenceTol: Double = 0.0
  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
   * Set the convergence tolerance. Default 0.001
   * convergenceTol is a condition which decides iteration termination.
   * The end of iteration is decided based on below logic.
   *
   *  - If the norm of the new solution vector is greater than 1, the diff of solution vectors
   *    is compared to relative tolerance which means normalizing by the norm of
   *    the new solution vector.
   *  - If the norm of the new solution vector is less than or equal to 1, the diff of solution
   *    vectors is compared to absolute tolerance which is not normalizing.
   *
   * Must be between 0.0 and 1.0 inclusively.
   */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * :: DeveloperApi ::
   * Runs gradient descent on the given training data.
    *
    * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GhandGradientDescent_breeze.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol)
    weights
  }

}

/**
 * :: DeveloperApi ::
 * Top-level method to run gradient descent.
 */
@DeveloperApi
object GhandGradientDescent_breeze extends Logging {
  /**
   * Run stochastic gradient descent (SGD) in parallel using mini batches.
   * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
   * in order to compute a gradient estimate.
   * Sampling, and averaging the subgradients over this subset is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data Input data for SGD. RDD of the set of data examples, each of
   *             the form (label, [feature values]).
   * @param gradient Gradient object (used to compute the gradient of the loss function of
   *                 one single data example)
   * @param updater Updater function to actually perform a gradient step in a given direction.
   * @param stepSize initial step size for the first step
   * @param numIterations number of iterations that SGD should be run.
   * @param regParam regularization parameter
   * @param miniBatchFraction fraction of the input data set that should be used for
   *                          one iteration of SGD. Default value 1.0.
   * @param convergenceTol Minibatch iteration will end before numIterations if the relative
   *                       difference between the current weight and the previous weight is less
   *                       than this value. In measuring convergence, L2 norm is calculated.
   *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the
   *         stochastic loss computed for every iteration.
   */
  def runMiniBatchSGD(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      stepSize: Double,
      numIterations: Int,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: Vector,
      convergenceTol: Double): (Vector, Array[Double]) = {

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    if (numIterations * miniBatchFraction < 1.0) {
      logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    /**
     * For the first iteration, the regVal will be initialized as sum of weight squares
     * if it's L2 updater; for L1 updater, the same logic is followed.
     */
    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) {
      logInfo(s"ghandCP=IterationId:${i}=BroadcastStartsTime:${System.currentTimeMillis()}")
      val bcWeights = data.context.broadcast(weights)
      logInfo(s"ghandCP=IterationId:${i}=BroadcastEndsTime:${System.currentTimeMillis()}")

      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val (weights_reduce, lossSum, miniBatchSize, numPartition, factorForSparseL2) = data.sample(false, miniBatchFraction, 42 + i)
          .treeAggregate(BDV.zeros[Double](n), 0.0, 0L, 0L, 1.0)(
            // zerovalue, loss, numberOfsampleused, numberOfPartition
            // (numberOfPartition also helps us to judge whether first seqOp
            seqOp = (c, v) => {
            // c: (weight_bar, loss, count_sample, count_partition, factor), v: (label, features)
              // weight_bar * factor is the real weight.
              if (c._4 == 0){
                  c._1 += bcWeights.value.asBreeze // += is overloaded
//                c._1 = c._1 + bcWeights.value.asBreeze // this is wrong, cuz you are assign things to a val
              }
              val thisIterStepSize = stepSize
              val transStepSize = thisIterStepSize / (1 - thisIterStepSize * regParam) / c._5
              val dotProduct = dot(v._2, Vectors.fromBreeze(c._1)) * c._5
              val labelScaled = 2 * v._1 - 1.0
              val local_loss = if (1.0 > labelScaled * dotProduct) {
                axpy((-labelScaled) * (-transStepSize), v._2, Vectors.fromBreeze(c._1))
                1.0 - labelScaled * dotProduct
              } else {
                0.0
              }
              // TODO: should add regularization here. ghand
              if (c._3 != 0 && c._3 % 5000 ==0){
                // really update weight via L2 regularization, otherwise there will be a numeric issue.
                // need learn_rate * regulartization <= 0.2
                // to avoid numeric issue.
                c._1 *= c._5 * (1 - thisIterStepSize * regParam)
                (c._1, c._2 + local_loss, c._3 + 1, 1L, 1.0)
              }
              else {
                // lazy update
                (c._1, c._2 + local_loss, c._3 + 1, 1L, (1 - thisIterStepSize * regParam) * c._5)
              }
//              (c._1, c._2 + local_loss, c._3 + 1, 1L, (1 - thisIterStepSize * regParam) * c._5)

            },
          combOp = (c1, c2) => {
            // user code cannot get into these function, since this function will be transfered
            // to resulthandler, they are very sensitive to user Functions. A Spark problem.
            // c: (grad, loss, count)
            // cuz this is no x = x + y in JBLAS
            // w_bar * c_t + w_bar*c_t. c_t is not useful anymore, so it is set as 0.
            logInfo(s"ghand=factorForSparseL2:${c1._5}:${c2._5}")
            // avoid renew breeze objects
            c1._1 *= c1._5
            c1._1 += c2._1 * c2._5
            (c1._1, c1._2 + c2._2, c1._3 + c2._3, c1._4 + c2._4, 1.0)
            // the last part should be 1. Not zero.
          }
        )
      logInfo(s"ghandCP=IterationId:${i}=DestroyBroadcastStartsTime:${System.currentTimeMillis()}")
      bcWeights.destroy(blocking = false)
      logInfo(s"ghandCP=IterationId:${i}=DestroyBroadcastEndsTime:${System.currentTimeMillis()}")

      if (miniBatchSize > 0) {
        /**
         * lossSum is computed using the weights from the previous iteration
         * and regVal is the regularization value computed in the previous iteration as well.
         */
        stochasticLossHistory += lossSum / miniBatchSize + regVal

        val weights_avg = weights_reduce / numPartition.toDouble
        weights = Vectors.fromBreeze(weights_avg)
        // this is very important, if not update, the weights will always be zero.

        val norm_value = brzNorm(weights_avg, 2.0)
        regVal = 0.5 * regParam * norm_value * norm_value

        logInfo(s"ghandCP=IterationId:${i}=JudgeConvergeStartsTime:${System.currentTimeMillis()}")
        previousWeights = currentWeights
        currentWeights = Some(Vectors.fromBreeze(weights_avg))

        if (previousWeights != None && currentWeights != None) {
          converged = isConverged(previousWeights.get,
            currentWeights.get, convergenceTol)
        }
        logInfo(s"ghandCP=IterationId:${i}=JudgeConvergeEndsTime:${System.currentTimeMillis()}")

        // compute the training loss here.
        if (TaskContext.isDebug) {
          val trainLoss_start_ts = System.currentTimeMillis()
          val train_loss = data.map(x => math.max(0, 1.0 - (2.0 * x._1 - 1.0) * dot(x._2, weights)))
            .reduce((x, y) => x + y)
          logInfo(s"ghandTrainLoss=IterationId:${i}=" +
            s"EpochID:${i * miniBatchFraction}=" +
            s"startLossTime:${trainLoss_start_ts}=" +
            s"EndLossTime:${System.currentTimeMillis()}=" +
            s"trainLoss:${(train_loss) / numExamples}")
          // this is the right way of computing regval in default MLLib
          val breeze_weight = weights.asBreeze.toDenseVector
          val norm_value_debug = brzNorm(breeze_weight, 2)
          logInfo(s"ghand=weightNorm:${norm_value_debug}")
        }

      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }

    logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)

  }

  /**
   * Alias of `runMiniBatchSGD` with convergenceTol set to default value of 0.001.
   */
  def runMiniBatchSGD(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      stepSize: Double,
      numIterations: Int,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: Vector): (Vector, Array[Double]) =
    GradientDescent.runMiniBatchSGD(data, gradient, updater, stepSize, numIterations,
                                    regParam, miniBatchFraction, initialWeights, 0.001)


  private def isConverged(
      previousWeights: Vector,
      currentWeights: Vector,
      convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = brzNorm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(brzNorm(currentBDV), 1.0)
  }

}