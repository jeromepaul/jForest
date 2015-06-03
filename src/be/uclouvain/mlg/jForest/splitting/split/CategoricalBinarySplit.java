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

import java.util.Arrays;

/**
 * An instance of this class contains the splitting information
 * to separate instances into two child nodes
 * based two partitions of the values of a categorical variable (like CART trees)
 */
public class CategoricalBinarySplit extends UnivariateSplit{

	private double[] catLeft;
	
	/**
	 * Initializes a binary split on the categorical variable <code>var</code>.
     * @param var the column index of the variable in the full dataset
     * @param index the value of the metric optimized by the split (<i>e.g.</i> Gini)
	 * @param catLeft a subset of the levels of feature <code>var</code>.
	 *        The two child nodes respectively contain instances with <code>x[var] in catLeft</code> and <code>x[var] not in catLeft</code>.
	 */
	public CategoricalBinarySplit(int var, double index, double[] catLeft){
		super(var,index);
		this.catLeft = catLeft;
	}
	
	@Override
	public int getChildIdFor(double[] sampleFeatures) {
		return getChildIdFor(sampleFeatures[getVarId()]);
	}
	
	@Override
	public int getChildIdFor(double value) {
		for(double d : catLeft){
			if(value == d) return 0;
		}
		return 1;
	}
	

	@Override
	public String toString() {
		return "Split on cat var "+getVarId()+" with categories going to left "+Arrays.toString(catLeft)+" and index "+getIndex();
	}
}
