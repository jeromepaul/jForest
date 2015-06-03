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
 * Aggregation of the internal (split) importance of a forest using a mean.
 */
public class AverageSplitIndex implements InternalImportanceIF {
	
	private double[] importances;
	private int count;
	
	/**
	 * @param nfeat the number of variables in the dataset
	 */
	public AverageSplitIndex(int nfeat){
		importances = new double[nfeat];
		count = 0;
	}
	
	@Override
	public double[] getImportances() {
		double[] res = new double[importances.length];
		for(int i = 0; i < res.length; i++) res[i] = importances[i] / count;
		return res;
	}
	
	@Override
	public void addImportanceOfSplit(Splitter split){
		for(int v : split.getVariableUsedToSplit()){
			importances[v] += split.getImportanceIndex();
		}
		count++;
	}
	
	@Override
	public String toString(){
		return "Mean importances : "+Arrays.toString(getImportances()); //+"\nVariable count : "+Arrays.toString(usedVariableCount);
	}
	
}
