/*
jForest
Copyright © 2010-2015, Université catholique de Louvain, Belgium - UCL
All rights reserved.

This file is part of the jForest library.

jForest has been developed by Jérôme Paul
(Machine Learning Group (MLG) - Institute of Information and Communication
Technologies, Electronics and Applied Mathematics (ICTEAM)) for the
Université catholique de Louvain (UCL). jForest is a general framework for
Machine Learning. It implements tree ensemble based classification methods.
It is designed to be very modular and allows easy tuning and modification of
the tree induction, classification criterion and feature importance index.

jForest is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

jForest is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with jForest.  If not, see <http://www.gnu.org/licenses/>.
*/

package be.uclouvain.mlg.jForest.importance.external.permutation;

import java.util.Random;

import org.apache.commons.math3.stat.inference.ChiSquareTest;

import be.uclouvain.mlg.jForest.data.Data;
import be.uclouvain.mlg.jForest.importance.external.permutation.indices.PermutationIndexWithDimIF;

/**
 * Given a contingency table, computes a chi^2 test on the sum over all trees of the contingency tables for each variable.
 * See J. Paul and P. Dupont, <a href="http://dx.doi.org/10.1016/j.neucom.2014.07.067" target="_blank">Inferring statistically significant features from random forests</a>, Neurocomputing, Volume 150, Part B, 20 February 2015, pp. 471-480, ISSN 0925-2312.
 */
public class ChisqOnTwoConfusionTables extends PermImportance<double[], long[][]> {
	
	private long[][][] sum;
	private int nfeat;
	private boolean returnPval;

	/**
     * Initialize an object that aggregates contingency table with a sum followed by a chi^2 test
     * @param d the dataset
     * @param rnd a random number generator used to create permutations of the OOB
     * @param permutationIndex an object that computes a statistic for one variable in a particular tree
     * @param pval should the computed importance be <i>p</i>-values or the chi^2 statistic ? 
     */
	public ChisqOnTwoConfusionTables(Data d, Random rnd, PermutationIndexWithDimIF<long[][]> permutationIndex, boolean pval) {
		super(d, rnd, permutationIndex);
		int[] dim = permutationIndex.getDim();
		this.nfeat = d.getP();
		sum = new long[nfeat][dim[0]][dim[1]];
		this.returnPval = pval;
	}

	@Override
	public double[] getImportances() {
		double[] res = new double[nfeat];
		ChiSquareTest x = new ChiSquareTest();
		
		for(int i = 0; i < nfeat; i++){
			
			if(returnPval){
				res[i] = x.chiSquareTest(sum[i]);
			}
			else{
				res[i] = x.chiSquare(sum[i]);
			}
		}
		return res;
	}

	@Override
	protected void addStatOfVar(int v, long[][] index, int treeId) {
		if(index.length != sum[0].length || index[0].length != sum[0][0].length) throw new RuntimeException("index dim is different from sum");
		
		for(int i = 0; i < index.length; i++){
			for(int j = 0; j < index[0].length; j++){
				sum[v][i][j] += index[i][j];
			}
		}
	}

}
