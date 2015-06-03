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

package be.uclouvain.mlg.jForest.importance.external.permutation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;
import java.util.TreeSet;

import be.uclouvain.mlg.jForest.data.Data;
import be.uclouvain.mlg.jForest.importance.external.ExternalImportanceIF;
import be.uclouvain.mlg.jForest.importance.external.permutation.indices.PermutationIndexIF;
import be.uclouvain.mlg.jForest.tree.Tree;
import be.uclouvain.mlg.jForest.tree.TreeIF;

/**
 * This class defines feature importance as a comparison between two statistics
 * extracted from the decision trees and their corresponding OOB
 * (<i>e.g.</i> the predictive performance of a tree on its OOB).
 * The first statistic is computed from the original OOB,
 * the second one is computed from a permuted version of the OOB, with a particular variable randomly shuffled. 
 */
public abstract class PermImportance<E,I> implements ExternalImportanceIF<E> {
	
	private Data d;
	private Random rnd;
	private PermutationIndexIF<I> permutationIndex;
	protected int treeCount;
	
	/**
	 * This is not intended to be called directly but it has to be called by extending classes.
	 * @param d the dataset
	 * @param rnd a random number generator used to create permutations of the OOB
	 * @param permutationIndex an object that computes a statistic for one variable in a particular tree
	 */
	public PermImportance(Data d, Random rnd, PermutationIndexIF<I> permutationIndex){
		this.d = d;
		this.rnd = rnd;
		this.permutationIndex = permutationIndex;
		treeCount = 0;
	}
	
	/**
	 * Adds the statistic of variable <code>v</code> computed on one tree.
	 * This should be called for each variable on each tree. 
	 * @param v is the column index of the variable in the full dataset
	 * @param stat is the value of the statistic computed on one tree with and without permuting variable <code>v</code>
	 * @param treeId is the id of the current tree
	 */
	protected abstract void addStatOfVar(int v, I stat, int treeId);
	
	
	@Override
	public void addImportanceOfTree(TreeIF tree, int treeId) {
		
		TreeSet<Integer> varInTree = new TreeSet<Integer>();
		for(int i : getVariablesToTest(tree)){
			varInTree.add(i);
		}
		
		for(int i = 0; i < d.getP(); i++){ // for each variable
			double[][] oob = getOob(tree.getOob());
			
			double[][] permOob = null;
			if(varInTree.contains(i)) permOob = getPermutedOOB(i, tree.getOob());
			
			permutationIndex.reInit();
			
			for(int s = 0; s < tree.getOob().length; s++){ // for each OOB sample // tree.getOob()[s] is the true index in data of sample oob[s]
				int pred = tree.predict(oob[s]);
				if(varInTree.contains(i)){
					int predPerm = tree.predict(permOob[s]);
					permutationIndex.addPoint(pred, predPerm, d.getLabels()[tree.getOob()[s]], true, tree.getOob()[s]);
				}
				else{
					permutationIndex.addPoint(pred, pred, d.getLabels()[tree.getOob()[s]], false, tree.getOob()[s]);
				}
			}
			
			addStatOfVar(i, permutationIndex.getIndex(),treeId);
			
		}
		
		treeCount++;
	}

	/**
	 * @param tree is a tree of the forest
	 * @return the variables used to build that tree
	 */
	private LinkedList<Integer> getVariablesToTest(TreeIF tree){

		LinkedList<Integer> res = new LinkedList<Integer>();

		double[] varCount = ((Tree) tree).getInTreeVariableCount().getImportances();

		for(int i = 0; i < varCount.length; i++){
			if(varCount[i] > 0){
				res.add(i);
			}
		}

		return res;
	}

	/**
	 * @param oob is an array containing the indices of samples in oob
	 * @return an |oob|x|nfeat| array in which each row corresponds to a data point
	 */
	private double[][] getOob(int[] oob){

		double[][] res = new double[oob.length][];
		for(int i = 0; i < oob.length; i++) res[i] = d.getData()[oob[i]];

		return res;
	}

	/**
	 * @param permFeatId feature to permute
	 * @param oob is an array containing the indices of the oob samples in the dataset
	 * @return the data about the oob with feature permFeatId permuted
	 */
	private double[][] getPermutedOOB(int permFeatId, int[] oob){
		double[][] permuted = new double[oob.length][d.getP()];

		for(int i = 0; i < permuted.length; i++){
			for(int j = 0; j < permuted[i].length; j++){
				permuted[i][j] = d.getData()[oob[i]][j];
			}
		}

		ArrayList<Integer> shuffledId = new ArrayList<Integer>(oob.length);
		for(int i : oob) shuffledId.add(i);
		Collections.shuffle(shuffledId,rnd);

		for(int i = 0; i < permuted.length; i++) permuted[i][permFeatId] = d.getData()[shuffledId.get(i)][permFeatId];

		return permuted;
	}

}
