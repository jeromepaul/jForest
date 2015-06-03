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
import java.util.Arrays;

import be.uclouvain.mlg.jForest.splitting.split.UnivariateSplit;

/**
 * Abstract definition of objects that perform binary splits.
 */
public abstract class Binary1FeatureSplitter extends Splitter {

	private boolean isCatSplit;

	@Override
	public void findBestSplit() {
		bestImpIndex = -1; 
		int bestVar = -1;
		for(int f : mtryVars){
			// in case of tie, the first variable is kept. However the order of those variable can be random in mtryVars.
			UnivariateSplit tmp = getSplitAndIndex(f);
			if(tmp.getIndex() > bestImpIndex){
				bestImpIndex = tmp.getIndex();
				bestVar = f;
				isCatSplit = d.getIsCat()[f];
				howToSplit = tmp;
			}
		}
		nbChildren = 2;
		varUsedInSplit = new int[]{bestVar};
		int[][] repartition = getRepartition();
		if(repartition[0].length == 0 || repartition[1].length == 0) nbChildren = 1; // happens if we cannot separate the data
	}

	/**
	 * This method should be overwritten to compute the best split for a particular feature.
	 * @param f a feature of <code>tree.getData()</code> for which we compute the best split on only points in <code>tree.getSampleIds()</code>
	 * @return an array of two elements
	 *     <ol>
	 *         <li>the value leading to the best split</li>
     *         <li>the corresponding metric value</li>
     *     </ol> 
	 * @pre The current object is initialized.
	 *      This should only be called with features in <code>tree.getMtryVars()</code>
	 */
	protected abstract UnivariateSplit getSplitAndIndex(int f);

	@Override
	public String toString(){
		return "Splitting on "+howToSplit+ " (giniDrop = "+getImportanceIndex()+", mtryVars = "+Arrays.toString(mtryVars)+", nbChildren = "+nbChildren+", isCatSplit = "+isCatSplit+")";
	}

}

