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

package be.uclouvain.mlg.jForest.splitting.split;

/**
 * An instance of this class is used in each node to stock the splitting information
 * <i>i.e.</i> the rule to classify new samples and the value of the metric optimized while training (<i>e.g.</i> gini).  
 */
public abstract class UnivariateSplit {

	private double index;
	private int var;
	
	/**
	 * Initializes a univariate split on variable <code>var</code> which yielded a value of <code>index</code>
	 * @param var the column index of the variable in the full dataset
	 * @param index the value of the metric optimized by the split (<i>e.g.</i> Gini)
	 */
	public UnivariateSplit(int var, double index){
		this.index = index;
		this.var = var;
	}
	
	/**
	 * Computes in which child node an instance falls into
	 * @param x a <i>p</i>-dimensional sample to percolate down to a child node
	 * @return the index in <code>[0 ; this.getNbChildren()[</code> of the child in which the training sample falls into
	 */
	public abstract int getChildIdFor(double[] x);
	
	/**
	 * Computes in which child node an instance falls into according the value of the feature of interest 
	 * @param v the value of feature <code>var</code> (see constructor) of a sample to percolate down to a child node
	 * @return the index in <code>[0 ; this.getNbChildren()[</code> of the child in which the training sample falls into
	 */
	public abstract int getChildIdFor(double v);
	
	/**
	 * @param index the value of the metric optimized by the split (<i>e.g.</i> Gini)
	 */
	public void setIndex(double index){
		this.index = index;
	}
	
	/**
	 * @return the value of the metric optimized by the split (<i>e.g.</i> Gini)
	 */
	public double getIndex() {
		return index;
	}
	
	/**
	 * @return the column index of the splitting variable in the full dataset
	 */
	public int getVarId(){
		return var;
	}

	@Override
	public String toString() {
		return "SplitAndIndex : var = "+var+", index = "+index;
	}
}
