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
 * For a variable in a tree : computes the decrease in classification accuracy between the OBB and permuted OOB
 */
public class AccuracyDrop implements PermutationIndexIF<Double> {
	
	private int count, errPred, errPermPred;
	
	public AccuracyDrop(){
		reInit();
	}

	@Override
	public void reInit() {
		count = 0;
		errPred = 0;
		errPermPred = 0;
	}

	@Override
	public void addPoint(int pred, int permPred, int label, boolean isVariableInTree, int pointIndex) {
		// rem: if !isVarInTree then pred == permPred --> nothing changes
		if(pred != label) errPred++;
		if(permPred != label) errPermPred++;
		count++;
	}

	@Override
	public Double getIndex() {
		if(count == 0) return 0.0;
		return ((double)(errPermPred - errPred)) / count;
	}
}
