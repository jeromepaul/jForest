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

package be.uclouvain.mlg.jForest.importance.external.permutation.indices;

/**
 * For a variable in a tree :
 * Computes a contingency table with two vectorized confusion matrices:
 * one for the original OOB and the other for a permuted OOB.
 *  
 *  See J. Paul and P. Dupont,
 *  <a href="http://dx.doi.org/10.1016/j.neucom.2014.07.067" target="_blank">Inferring statistically significant features from random forests</a>,
 *  Neurocomputing, Volume 150, Part B, 20 February 2015, pp. 471-480, ISSN 0925-2312.
 */
public class TwoConfusionMatrices implements PermutationIndexWithDimIF<long[][]> {
	
	//     \vote of the tree
	// true | 0 1
	//     0| a  b
	//     1| c  d
	//
	// ---> [a,b,c,d]
	private long[] nonPerm, perm;
	private int nCells, nLabelLevels;
	
	/**
	 * @param nClasses the number of classes in the dataset
	 */
	public TwoConfusionMatrices(int nClasses){
		this.nLabelLevels = nClasses;
		nCells = nClasses*nClasses;
		nonPerm = new long[nCells];
		perm = new long[nCells];
		reInit();
	}

	@Override
	public void reInit() {
		for(int i = 0;  i < nCells; i++){
			nonPerm[i] = 0;
			perm[i] = 0;
		}
	}

	@Override
	public void addPoint(int pred, int permPred, int label, boolean isVariableInTree, int pointIndex) {
			nonPerm[pred + nLabelLevels*label]++;
			perm[permPred + nLabelLevels*label]++;
	}

	@Override
	public long[][] getIndex() {
		long[][] res = new long[2][];
		res[0] = nonPerm;
		res[1] = perm;
		return res;
	}

	@Override
	public int[] getDim() {
		return new int[]{2,nonPerm.length};
	}
}
