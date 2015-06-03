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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;

import be.uclouvain.mlg.jForest.splitting.index.Gini;
import be.uclouvain.mlg.jForest.splitting.split.CategoricalBinarySplit;
import be.uclouvain.mlg.jForest.splitting.split.ContinuousBinarySplit;
import be.uclouvain.mlg.jForest.splitting.split.UnivariateSplit;
import be.uclouvain.mlg.jForest.splitting.tools.CatCount;

/**
 * CART like splits based on the drop in Gini.
 */
public class CARTSplitter extends Binary1FeatureSplitter {
	
	@Override
	protected UnivariateSplit getSplitAndIndex(int f) {
		
		int[] classes = d.getLabelLvls();
		
		UnivariateSplit res = new ContinuousBinarySplit(-1, Double.NEGATIVE_INFINITY, Double.NaN);

		if(d.getIsCat()[f]){ // runs in O(nlevels * log(nlevels) + n)
			for(int c : classes){
				res = computeCatSplit(f, c, res); // res is updated only if new result is better than the one passed to the method
			}
		}
		else{ // runs in O(n * log n)
			// get the values of in-bag samples for feature f and sort them
			ValLabel[] sortedFeat = new ValLabel[sampleIds.length];
			for(int i = 0; i < sampleIds.length; i++) sortedFeat[i] = new ValLabel(d.getData()[sampleIds[i]][f],d.getLabels()[sampleIds[i]]);
			Arrays.sort(sortedFeat);

			// all the different gini will be computed in two traversal of the sortedFeatList
			Gini gCur = new Gini(d.getNumberOfClasses());
			Gini gLeft = new Gini(d.getNumberOfClasses());
			Gini gRight = new Gini(d.getNumberOfClasses());

			for(ValLabel vl : sortedFeat){
				gCur.addLabel(vl.label);
				gRight.addLabel(vl.label);
			}

			for(int i = 0; i < sortedFeat.length - 1 ; i++){
				double valL = sortedFeat[i].val;
				double valR = sortedFeat[i+1].val;
				gLeft.addLabel(sortedFeat[i].label);
				gRight.removeLabel(sortedFeat[i].label);
				double tmp = (valL == valR)?0:Gini.getGiniDrop(gCur, gLeft, gRight);
				if(tmp > res.getIndex()){
					res = new ContinuousBinarySplit(f, tmp, (valL + valR) / 2);
				}
			}
		}
		
		return res;

	}
	
	/**
	 * To create one VS all classification problem
	 * Defines class of interest as 1 and all other classes as 0
	 */
	private static int getFakeLabel(int label, int classOfInterest){
		return (label == classOfInterest)?1:0;
	}
	
	private UnivariateSplit computeCatSplit(int f, int classOfInterest, UnivariateSplit curRes) {
		Gini gCur = new Gini(2);
		Gini gLeft = new Gini(2);
		Gini gRight = new Gini(2);
		
		for(int i : sampleIds){
			int l = getFakeLabel(d.getLabels()[i],classOfInterest);
			gCur.addLabel(l);
			gRight.addLabel(l);
		}
		
		/*
		 * Computational shortcut :
		 * For binary classification problems, with 2 classes, the tree can order the categories by class probability for one of the classes.
		 * Then, the optimal split is one of the nlevels-1 splits from the ordered list.
		 * (from http://www.mathworks.nl/help/stats/splitting-categorical-predictors-for-multiclass-classification.html (consulted in June 2013))
		 * 
		 * Breiman, L., J. H. Friedman, R. A. Olshen, and C. J. Stone. Classification and Regression Trees. Chapman & Hall, Boca Raton, 1993.
		 */
		
		Hashtable<Double, CatCount> catCountsHT = new Hashtable<Double,CatCount>();
		for(double lvl : d.getLevelsOfCatVar(f)) catCountsHT.put(lvl, new CatCount(lvl, 1, 0)); // O(nlevels)
		for(int i : sampleIds){
			int fakeLabel = getFakeLabel(d.getLabels()[i], classOfInterest);
			catCountsHT.get(d.getData()[i][f]).add(fakeLabel); // O(n)
		}
		
		CatCount[] catCounts = new CatCount[d.getLevelsOfCatVar(f).length];
		for(int i = 0; i < catCounts.length; i++) catCounts[i] = catCountsHT.get(d.getLevelsOfCatVar(f)[i]);
		
		Arrays.sort(catCounts); // O(nlevels * log(nlevels))
		
		ArrayList<Double> curCatsLeft = new ArrayList<Double>(catCounts.length);
		for(int i = 0; i < catCounts.length-1; i++){ // O(nlevels)
			curCatsLeft.add(catCounts[i].getLevel());
			gLeft.addLabel(catCounts[i].getC0(), catCounts[i].getN0());
			gLeft.addLabel(catCounts[i].getC1(), catCounts[i].getN1());
			gRight.removeLabel(catCounts[i].getC0(), catCounts[i].getN0());
			gRight.removeLabel(catCounts[i].getC1(), catCounts[i].getN1());
			
			double tmp = Gini.getGiniDrop(gCur, gLeft, gRight);
			if(tmp > curRes.getIndex()){
				double[] tmp2 = new double[curCatsLeft.size()];
				for(int j = 0; j < tmp2.length; j++) tmp2[j] = curCatsLeft.get(j);
				curRes = new CategoricalBinarySplit(f, tmp, tmp2);
			}
		}
		return curRes;
	}

	private class ValLabel implements Comparable<ValLabel> {
		public final double val;
		public final int label;

		public ValLabel(double val, int label){
			this.val = val;
			this.label = label;
		}

		@Override
		public int compareTo(ValLabel o) {
			return Double.compare(this.val, o.val);
		}

		@Override
		public String toString() {
			return "("+val+","+label+")";
		}
	}
}

