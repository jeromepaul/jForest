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

package be.uclouvain.mlg.jForest.tree;

/**
 * Interface of a tree to be grown in the forest
 */
public interface TreeIF {
	
	/**
	 * Grow the tree from a set of learning samples from at least two classes.
	 */
	public void grow();
	
	/**
	 * Predicts the label of a <i>p</i>-dimensional sample.
	 * @param x the feature vector of a sample to be predicted.
	 * @return the label of the prediction
	 */
	public int predict(double[] x);
	
	
	/**
	 * Computes the depth of the tree.
	 * The root node is at depth 0.
	 * A decision stump has a depth of 1.  
	 * @return tree depth
	 */
	public int getDepth();
	
	/**
	 * The tree might be grown from a subset of the whole dataset.
	 * The data points not used during the tree induction form the <i>out-of-bag (OOB)</i>.
	 * This methods returns the indices of OOB samples in an array.
	 * @return the indices of the OOB samples.
	 */
	public int[] getOob();
	
}
