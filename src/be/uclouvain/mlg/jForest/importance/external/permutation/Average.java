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

import be.uclouvain.mlg.jForest.data.Data;
import be.uclouvain.mlg.jForest.importance.external.permutation.indices.PermutationIndexIF;

/**
 * Given a scalar statistic, computes the average statistic for each variable over all trees  
 */
public class Average extends PermImportance<double[],Double> {
	
	private double[] sumOfPermutationIndex;
	
	/**
	 * Initialize an object that aggregates statistics with a mean
	 * @param d the dataset
     * @param rnd a random number generator used to create permutations of the OOB
     * @param permutationIndex an object that computes a statistic for one variable in a particular tree
	 */
	public Average(Data d, Random rnd, PermutationIndexIF<Double> permutationIndex){
		super(d,rnd,permutationIndex);
		sumOfPermutationIndex = new double[d.getP()];
	}

	@Override
	public double[] getImportances() {
		double[] res = sumOfPermutationIndex.clone();
		for(int i = 0; i < res.length; i++) res[i] /= treeCount;
		return res;
	}

	@Override
	protected void addStatOfVar(int v, Double index, int treeId) {
		sumOfPermutationIndex[v] += index;
	}

}
