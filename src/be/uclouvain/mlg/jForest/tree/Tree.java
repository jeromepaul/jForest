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

package be.uclouvain.mlg.jForest.tree;

import java.util.Arrays;

import be.uclouvain.mlg.jForest.data.Data;
import be.uclouvain.mlg.jForest.importance.internal.InternalImportanceIF;
import be.uclouvain.mlg.jForest.sampler.RandomSampler;
import be.uclouvain.mlg.jForest.splitting.Splitter;

/**
 * Generic definition of a decision tree
 */
public class Tree implements TreeIF{
	
	private Data d;
	private TreeIF[] subTrees;
	private int[] inTree, oob, mtryVars;
	private RandomSampler mtrySampler;
	private Class<Splitter> splittingCriterionClass;
	private Splitter splittingCriterion;
	private InternalImportanceIF inImpOfForest;
	private InternalImportanceIF varCount;
	private final int maxDepth;
	
	/**
	 * Initializes a tree object with the following parameters 
	 * @param mtrySampler a sampler used to randomly choose the candidate variables in each split
	 * @param splittingCriterionClass the class according to which splits are decided
	 * @param d the full dataset on which the ensemble is grown
	 * @param inTree the indices of the samples of <code>d</code> from which this tree is grown
	 * @param oob the indices of the samples of <code>d</code> that are not used to grow this tree
	 * @param inImpOfForest an aggregator of variable importance, computed from the splitting criteria.
	 *                      This object should be common to all trees in the ensemble.
	 * @param variableCount an object that records the number of times variables are used for splitting in the current tree.
	 *                      This object should be different for each tree.
	 * @param maxDepth the maximal depth of this tree. A negative value will cause to fully grow the tree.
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tree(RandomSampler mtrySampler, Class splittingCriterionClass, Data d, int[] inTree, int[] oob, InternalImportanceIF inImpOfForest, InternalImportanceIF variableCount, int maxDepth){
		this.d = d;
		this.inTree = inTree;
		this.oob = oob;
		this.mtrySampler = mtrySampler;
		this.splittingCriterionClass = splittingCriterionClass;
		this.inImpOfForest = inImpOfForest;
		this.varCount = variableCount;
		this.maxDepth = maxDepth;
	}
	
	/**
	 * Recursively grows the tree until one of those conditions is met:
	 * <ul>
	 *     <li>samples cannot be separated</li>
	 *     <li>all samples are correctly classified</li>
	 *     <li>maximal depth is reached.</li>
	 * </ul>
	 * In addition, the internal importance is updated in inImp object (see constructor).
	 */
	@Override
	public void grow(){
		mtryVars = mtrySampler.getSample(d.getColRange())[0];
		
		try {
			splittingCriterion = splittingCriterionClass.newInstance();
		} catch (InstantiationException | IllegalAccessException e) {
			e.printStackTrace();
		}
		
		
		if(maxDepth == 0){ // create one leaf node to perform classification
			subTrees = new TreeIF[1];
		}
		else{
			splittingCriterion.init(this);
			splittingCriterion.findBestSplit();
			subTrees = new TreeIF[splittingCriterion.getNbChildren()];
		}
		
		if(subTrees.length == 1){
			// cannot split the data or maximal depth reached --> one leaf node
			int[] majorityClasses = d.getMajorityClass(inTree);
			subTrees[0] = new Leaf(inTree,oob,majorityClasses[mtrySampler.getRandom().nextInt(majorityClasses.length)]);
		}
		else{
			inImpOfForest.addImportanceOfSplit(splittingCriterion); // update the internal importance of variables for all the forest
			varCount.addImportanceOfSplit(splittingCriterion); // update internal variable counts for the current tree only
			
			int[][] repartition = splittingCriterion.getRepartition();
			
			for(int i = 0; i < subTrees.length; i++){
				if(d.getNumberOfClassesIn(repartition[i]) == 1){ // one class --> one leaf
					subTrees[i] = new Leaf(repartition[i],oob,d.getLabels()[repartition[i][0]]);
				}
				else{
					subTrees[i] = new Tree(mtrySampler, splittingCriterionClass, d, repartition[i], oob,inImpOfForest,varCount,maxDepth-1);
				}
				subTrees[i].grow();
			}
		}
	}
	
	/**
	 * @param x a sample to be predicted (in <i>p</i> dimensions)
	 * @return the estimated class label of x
	 */
	@Override
	public int predict(double[] x){
		if(subTrees.length == 1) return subTrees[0].predict(x); // This happens when fixing a maximal depth
		return subTrees[splittingCriterion.getChildIdFor(x)].predict(x);
	}
	
	/**
	 * @return the aggregator which counts variables used in the current tree
	 */
	public InternalImportanceIF getInTreeVariableCount(){
		return varCount;
	}
	
	@Override
	public int[] getOob(){
		return oob;
	}
	
	@Override
	public int getDepth() {
		int tmp = Integer.MIN_VALUE;
		for(TreeIF t : subTrees){
			int maxSubDepth = t.getDepth(); 
			if(maxSubDepth > tmp) tmp = maxSubDepth; 
		}
		if(subTrees.length == 1) return tmp; // this is an artificial leaf (created because no split was found or max depth was reached)
		return tmp + 1;
	}
	
	@Override
	public String toString(){
		String res = "In tree : " + Arrays.toString(inTree) + "\n";
		res += "OOB : "+Arrays.toString(getOob())+"\n"; 
		res += "Splitting Criterion : "+splittingCriterion + "\n\n";
		for(TreeIF t : subTrees) res += t.toString();
		return res;
	}

	/**
	 * @return the full data matrix on which the ensemble is grown
	 */
	public Data getData() {
		return d;
	}

	/**
	 * @return the indices of the samples used to grow this tree
	 */
	public int[] getSampleIds() {
		return inTree;
	}

	/**
	 * @return the indices of the candidate variables considered in the root split of this tree 
	 */
	public int[] getMtryVars() {
		return mtryVars;
	}
	
	/**
	 * @return the sampler object used to randomly select the candidate variables in each split
	 */
	public RandomSampler getMtrySampler() {
		return mtrySampler;
	}

}
