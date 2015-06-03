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

import be.uclouvain.mlg.jForest.data.Data;
import be.uclouvain.mlg.jForest.splitting.split.UnivariateSplit;
import be.uclouvain.mlg.jForest.tree.Tree;

/**
 * This class is an abstract implementation of splitter objects 
 * that are used at each node to find the best split and then apply it to data
 */
public abstract class Splitter{
	
	protected Data d;
	protected int[] mtryVars;
	protected int[] sampleIds;
	protected int nbChildren;
	protected double bestImpIndex;
	protected int[] varUsedInSplit;
	protected UnivariateSplit howToSplit;
	
	private int[][] repartition;
	
	
	/**
	 * Initializes the current object which is created using the default constructor
	 * @param tree The split is operated in the root node of tree
	 */
	public void init(Tree tree){
		this.d = tree.getData();
		this.mtryVars = tree.getMtryVars();
		this.sampleIds = tree.getSampleIds();
		this.repartition = null;
		bestImpIndex = 0; // to be changed in findBestSplit
	}
	
	/**
	 * Learn the best split.
	 * @pre The current object must be initialized with <code>init(Tree tree)</code> before calling this method.
	 * @post When the call returns, the current object knows
	 * <ul>
	 *     <li>the best variable used for splitting</li>
	 *     <li>the number of children</li>
	 *     <li>the way to allocate a child for a given sample</li>
	 *     <li>the importance index of the split</li>
	 * </ul>
	 */
	public abstract void findBestSplit();
	
	/**
	 * @param sampleFeatures is the full vector of a data point
	 * @return the id in <i>[0 ; this.getNbChildren()[</i> of the child in which the training sample falls into
	 * @pre <code>findBestSplit()</code> was previously called on the current object
	 */
	public int getChildIdFor(double[] sampleFeatures){
		return howToSplit.getChildIdFor(sampleFeatures);
	}
	
	/**
	 * @param trainingSampleId index of a sample in <code>d</code>
	 * @return the index in <i>[0 ; this.getNbChildren()[</i> of the child in which training sample trainingSampleId falls into
	 * @pre <code>findBestSplit()</code> was previously called on the current object
	 */
	public int getChildIdFor(int trainingSampleId){
		return getChildIdFor(d.getData()[trainingSampleId]);
	}
	
	/**
	 * Computes the repartition of samples in subtrees.
	 * This is implemented with memoization.
	 * @return <code>res[i][]</code> is an array containing indices of samples falling into the <i>i</i>-th subtree
	 * @pre <code>findBestSplit()</code> was previously called on the current object
	 */
	public int[][] getRepartition(){
		if(repartition == null){
			// putting in an arrayList and then converting to int[][] has the same complexity
			int[] c = new int[nbChildren];
			for(int i : sampleIds) c[getChildIdFor(i)]++; // O(|sampleIds|)
			repartition = new int[nbChildren][];
			for(int i = 0; i < repartition.length; i++){ // O(|nbChildren|)
				repartition[i] = new int[c[i]];
				c[i] = 0;
			}
			for(int i : sampleIds){ // O(|sampleIds|)
				int tmp = getChildIdFor(i);
				repartition[tmp][c[tmp]] = i;
				c[tmp]++;
			}
		}
		
		return repartition;
	}
	
	/**
	 * @return the number of children in which samples will fall into according to the current split
	 * @pre <code>findBestSplit()</code> was previously called on the current object
	 */
	public int getNbChildren(){
		return nbChildren;
	}
	
	/**
	 * @return the importance index of the current split
	 * @pre <code>findBestSplit()</code> was previously called on the current object
	 */
	public double getImportanceIndex(){
		return bestImpIndex;
	}
	
	/**
	 * @return the index of the variables used to make the split
	 * @pre <code>findBestSplit()</code> was previously called on the current object
	 */
	public int[] getVariableUsedToSplit(){
		return varUsedInSplit;
	}
}
