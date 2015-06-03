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

package be.uclouvain.mlg.jForest.splitting;

import be.uclouvain.mlg.jForest.splitting.split.UnivariateSplit;

/**
 * RRF like splits.
 * Features that are not considered a priori relevant are penalized (their gini drop is multiplied by a constant between 0 and 1 (default 0.8).
 * See Deng, Houtao, and George Runger. "Feature selection via regularized trees." Neural Networks (IJCNN), The 2012 International Joint Conference on. IEEE, 2012.
 */
public class PriorKnowledgeCARTSplitter extends CARTSplitter {
	
	private static double coef = 0.8; // default value in RRF package cf. arXiv:1306.0237 and http://cran.r-project.org/web/packages/RRF/
	private static boolean[] isFavoredFeature;
	
	/**
	 * Sets the multiplicative constant to penalize features
	 * @param c a number between 0 and 1. The smaller <code>c</code> the more penalized the features
	 */
	public static void setCoef(double c){
		coef = c;
	}
	
	/**
	 * Sets the features to favor.
	 * Those variables which are favored are not penalized (the original gini drop value is kept).
	 * @param featToFavor <code>featToFavor[i]</code> iff variable <code>i</code> is not penalized.  
	 */
	public static void setFeatureToFavor(boolean[] featToFavor){
		isFavoredFeature = featToFavor;
	}
	
	@Override
	protected UnivariateSplit getSplitAndIndex(int feat){
		UnivariateSplit tmp = super.getSplitAndIndex(feat);
		if(!isFavoredFeature[feat])
			tmp.setIndex(coef * tmp.getIndex());
		return tmp;
	}
}
