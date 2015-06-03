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
 * General definition of an importance index that permutes variables in the OOB.
 * An object implementing this interface is intended to compute an importance value for one variable of one tree only.
 * @param <I> the type of the computed importance
 */
public interface PermutationIndexIF<I> {
	
	/**
	 * reinitializes the object (allowing not to create a new object for each variable in each tree)
	 */
	public void reInit();
	
	/**
	 * Add a new point for the computation of the current index value
	 * @param pred the label predicted on a point of the normal oob data
	 * @param permPred the label predicted on the same point but on the permuted oob data
	 * @param label the true label of the current point
	 * @param isVariableInTree iff the variable of interest appears in the current tree
	 * @param pointIndex the index of the sample in the dataset
	 */
	public void addPoint(int pred, int permPred, int label, boolean isVariableInTree, int pointIndex);
	
	/**
	 * @return the index computed on the points added via addPoint since the last call to reInit
	 */
	public I getIndex();
	
}
