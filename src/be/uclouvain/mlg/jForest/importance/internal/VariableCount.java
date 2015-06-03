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

package be.uclouvain.mlg.jForest.importance.internal;

import java.util.Arrays;

import be.uclouvain.mlg.jForest.splitting.Splitter;

/**
 * Aggregation of the internal (split) importance of a forest using a sum of occurrence.
 */
public class VariableCount implements InternalImportanceIF {
	
	private double[] usedVariableCount;
	
	/**
	 * @param nfeat the number of features in the dataset 
	 */
	public VariableCount(int nfeat){
		usedVariableCount = new double[nfeat];
	}
	
	@Override
	public double[] getImportances() {
		return usedVariableCount;
	}

	@Override
	public void addImportanceOfSplit(Splitter split) {
		for(int f : split.getVariableUsedToSplit()) usedVariableCount[f]++;
	}
	
	@Override
	public String toString(){
		return "Variable count : "+Arrays.toString(usedVariableCount);
	}
	
}
