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

import breeze.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._

/**
 * Utility methods for LDA.
 */
private[clustering] object LDAUtils {
  /**
   * Log Sum Exp with overflow protection using the identity:
   * For any a: $\log \sum_{n=1}^N \exp\{x_n\} = a + \log \sum_{n=1}^N \exp\{x_n - a\}$
   */
  private[clustering] def logSumExp(x: BDV[Double]): Double = {
    val a = max(x)
    a + log(sum(exp(x -:- a)))
  }

  private[clustering] def logVector(x: BDV[Double]): BDV[Double] = {
    x.map(a => log(a))
  }

  /**
   * For theta ~ Dir(alpha), computes E[log(theta)] given alpha. Currently the implementation
   * uses [[breeze.numerics.digamma]] which is accurate but expensive.
   */
  private[clustering] def dirichletExpectation(alpha: BDV[Double]): BDV[Double] = {
    digamma(alpha) - digamma(sum(alpha))
  }

  /**
   * Computes [[dirichletExpectation()]] row-wise, assuming each row of alpha are
   * Dirichlet parameters.
   */
  private[clustering] def dirichletExpectation(alpha: BDM[Double]): BDM[Double] = {
    val rowSum = sum(alpha(breeze.linalg.*, ::))
    val digAlpha = digamma(alpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }

  private[clustering] def dirichletExpectation(alpha: BDM[Double], ids: List[Int]): BDM[Double] = {
    val newAlpha = alpha.t(ids, ::).toDenseMatrix.t
    val rowSum = sum(alpha(breeze.linalg.*, ::))
    val digAlpha = digamma(newAlpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }

  private[clustering] def dirichletExpectation(alpha: BDM[Double], ids: List[Int], multiA1: Double,
                                               A3: Double, sumA1: Double, vocabSize: Int
                                              ): BDM[Double] = {
    val QAlpha = alpha.t(ids, ::).toDenseMatrix.t
    val newAlpha = QAlpha(::, breeze.linalg.*) * multiA1 + A3 * sumA1
    val rowSum = sum(alpha(breeze.linalg.*, ::)) * multiA1 + A3 * sumA1 * vocabSize
    val digAlpha = digamma(newAlpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }

  private[clustering] def dirichletExpectationTop(alpha: BDM[Double], existids: List[Int],
                                                  multiA1: Double, A3: Double, sumA1: Double,
                                                  vocabSize: Int,
                                                  topk: Int,
                                                  existElogBeta: BDM[Double],
                                                  deltaLambda: BDM[Double]): BDM[Double] = {
    val QAlpha = alpha.t(existids, ::).toDenseMatrix.t
    val newAlpha = QAlpha(::, breeze.linalg.*) * multiA1 + A3 * sumA1
    val digAlpha = digamma(newAlpha)

    val rowSum = sum(alpha(breeze.linalg.*, ::)) * multiA1 + A3 * sumA1 * vocabSize
    val digRowSum = digamma(rowSum)
    for (i <- 0 until rowSum.length) {
      System.out.print(s"-------------Sum--------------\n")
      System.out.print(s"sum of ${i}:${rowSum.apply(i)}\tdigamma Sum:${digRowSum.apply(i)}\n")
    }

    val result = digAlpha(::, breeze.linalg.*) - digRowSum

    val delta = deltaLambda(::, existids).toDenseMatrix
    val two = delta(::, breeze.linalg.*) / rowSum
    for (i <- 0 until two.cols) {
//      System.out.print(s"delta of ${i}:${delta(::, i).toString(toString)}\n")
//      System.out.print(s"two of ${i}:${two(::, i).toString()}\n")
      System.out.print(s"argTopk of ${i}:${argtopk(two(::, i), topk).toString()}\n")
      val index = argtopk(two(::, i), topk)
      for (j <- 0 until index.length) {
        val worth = alpha.apply(index(j), existids(i)) * multiA1 + A3 * sumA1
        val newworth = digamma(worth) - digRowSum.apply(index(j))
        existElogBeta.update(index(j), existids(i), newworth)
      }
    }

    System.out.print(s"result----------------------\n")
    System.out.print(s"${result}\n")
    System.out.print(s"existBeta---------------------\n")
    System.out.print(s"${existElogBeta(::, existids)}\n")

    existElogBeta
  }

}
