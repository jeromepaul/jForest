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
 * An instance of this class contains the splitting information
 * to separate instances into two child nodes based on a threshold on a continuous variable (like CART trees)
 */
public class ContinuousBinarySplit extends UnivariateSplit {

	private double splitVal;
	
	/**
	 * Initializes a binary split on the continuous variable <code>var</code> based on threshold <code>t</code>.
	 * @param var the column index of the variable in the full dataset
     * @param index the value of the metric optimized by the split (<i>e.g.</i> Gini)
	 * @param t the threshold value on <code>var</code>.
	 *          The two child nodes respectively contain instances with <code>x[var] < t</code> and <code>x[var] >= t</code>. 
	 */
	public ContinuousBinarySplit(int var, double index, double t) {
		super(var, index);
		this.splitVal = t;
	}

	@Override
	public int getChildIdFor(double[] sampleFeatures) {
		return getChildIdFor(sampleFeatures[getVarId()]);
	}
	
	@Override
	public int getChildIdFor(double value) {
		return (value < splitVal)?0:1;
	}

	@Override
	public String toString() {
		return "Split on cont var "+getVarId()+" with value "+splitVal+" and index "+getIndex();
	}
}
