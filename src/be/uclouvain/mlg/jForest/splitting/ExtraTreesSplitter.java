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
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;
import java.util.TreeSet;

import be.uclouvain.mlg.jForest.splitting.index.Gini;
import be.uclouvain.mlg.jForest.splitting.split.CategoricalBinarySplit;
import be.uclouvain.mlg.jForest.splitting.split.ContinuousBinarySplit;
import be.uclouvain.mlg.jForest.splitting.split.UnivariateSplit;
import be.uclouvain.mlg.jForest.tree.Tree;

/**
 * Extra trees like split as described in 
 * Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63(1), 3–42.
 */
public class ExtraTreesSplitter extends Binary1FeatureSplitter {

	protected Random rnd;

	@Override
	public void init(Tree tree) {
		super.init(tree);
		rnd = tree.getMtrySampler().getRandom();
	}

	@Override
	protected UnivariateSplit getSplitAndIndex(int f) {
		boolean isCat = d.getIsCat()[f];
		if(isCat) return getCatSplitAndIndex(f);
		return getContSplitAndIndex(f);
	}

	private UnivariateSplit getContSplitAndIndex(int f) {
		// max and min value between which we sample uniformly a threshold
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		Gini gCur = new Gini(d.getNumberOfClasses()); // gini of the current node
		UnivariateSplit split;
		
		for(int i : sampleIds){
			if(d.getData()[i][f] < min) min = d.getData()[i][f];
			if(d.getData()[i][f] > max) max = d.getData()[i][f];
			gCur.addLabel(d.getLabels()[i]);
		}
		
		split = new ContinuousBinarySplit(f, -1, rnd.nextDouble() * (max - min) + min);
		
		double giniDrop = computeGiniDropForVar(f, split, gCur);
		split.setIndex(giniDrop);
		
		return split;
	}

	private UnivariateSplit getCatSplitAndIndex(int f){
		// will contain the levels that appear in the current bag
		TreeSet<Double> ts = new TreeSet<Double>();
		Gini gCur = new Gini(d.getNumberOfClasses()); // gini of the current node
		UnivariateSplit split;
		
		for(int i : sampleIds){
			ts.add(d.getData()[i][f]);
			gCur.addLabel(d.getLabels()[i]);
		}
		
		// pick up a label at random
		LinkedList<Double> inBagLabels = new LinkedList<Double>();
		for(double d : ts) inBagLabels.add(d);
		
		double[] lvlsOfVar = d.getLevelsOfCatVar(f);
		ArrayList<Double> featLabels = new ArrayList<Double>(lvlsOfVar.length);
		for(double l : lvlsOfVar) featLabels.add(l);
		
		double[] catLeft = new double[0];
		while(!contains(catLeft, inBagLabels)){
			Collections.shuffle(featLabels,rnd);
			catLeft = new double[rnd.nextInt(featLabels.size())+1];
			for(int i = 0; i < catLeft.length; i++) catLeft[i] = featLabels.get(i);
		}
		
		split = new CategoricalBinarySplit(f, -1, catLeft);
		
		// compute gini drop
		double giniDrop = computeGiniDropForVar(f, split, gCur);
		split.setIndex(giniDrop);

		return split;
	}

	/**
	 * Computes the GiniDrop for variable f in current node
	 * @param f is the variable on which the gini drop is computed
	 * @param split is the way variable f is split
	 * @param gCur is the gini of the current node
	 * @return the drop in gini while splitting variable f according to split in current node
	 */
	private double computeGiniDropForVar(int f, UnivariateSplit split, Gini gCur){
		Gini gLeft = new Gini(d.getNumberOfClasses());
		Gini gRight = new Gini(d.getNumberOfClasses());
		for(int i : sampleIds){
			if(split.getChildIdFor(d.getData()[i][f]) == 0) gLeft.addLabel(d.getLabels()[i]);
			else gRight.addLabel(d.getLabels()[i]);
		}
		double giniDrop = Gini.getGiniDrop(gCur, gLeft, gRight);
		return giniDrop;
	}
	
	private static boolean contains(double[] goingToLeft, LinkedList<Double> inBagLabels){
		for(double l : goingToLeft){
			for(double in : inBagLabels){
				if(l == in) return true;
			}
		}
		return false;
	}
}
