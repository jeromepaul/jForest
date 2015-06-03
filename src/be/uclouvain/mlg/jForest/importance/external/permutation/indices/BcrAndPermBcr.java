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
 * For a variable in a tree : computes the balanced classification rate on the OBB and permuted OOB
 */
public class BcrAndPermBcr implements PermutationIndexIF<double[]>{

	private int nclass;
	
	/*
	 *  class | 0   1   2   ...
	 *  -----------------------
	 *  orig  | T0  T1  T2  ...  number of good pred for classes 0, 1 and 2 etc 
	 *  perm  | T0p T1p T2p ...  number of good pred for classes 0, 1 and 2 etc when x_j is permuted
	 *  tot   | #0  #1  #2  ...  number of points in class 0, 1 and 2 etc
	 */
	private int[][] counts;
	
	/**
	 * @param nclass the total number of classes in the dataset
	 */
	public BcrAndPermBcr(int nclass){
		this.nclass = nclass;
		counts = new int[3][nclass];
		// reInit(); // useless
	}
	
	@Override
	public void reInit() {
		for(int i = 0; i < counts.length; i++){
			for(int j = 0; j < counts[i].length; j++){
				counts[i][j] = 0;
			}
		}
	}

	@Override
	public void addPoint(int pred, int permPred, int label, boolean isVariableInTree, int pointIndex) {
		if(pred == label) counts[0][label]++;
		if(permPred == label) counts[1][label]++;
		counts[2][label]++;
	}

	@Override
	public double[] getIndex() {
		double bcr = 0;
		double permBcr = 0;
		for(int i = 0; i < nclass; i++){
			bcr += ((double) counts[0][i]) / counts[2][i];
			permBcr += ((double) counts[1][i]) / counts[2][i];
		}
		
		bcr /= nclass;
		permBcr /= nclass;
		
		return new double[]{bcr,permBcr};
	}

}
