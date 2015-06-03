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

package be.uclouvain.mlg.jForest.forest;

import be.uclouvain.mlg.jForest.data.Data;
import be.uclouvain.mlg.jForest.importance.external.ExternalImportanceIF;
import be.uclouvain.mlg.jForest.importance.internal.InternalImportanceIF;
import be.uclouvain.mlg.jForest.importance.internal.VariableCount;
import be.uclouvain.mlg.jForest.sampler.RandomSampler;
import be.uclouvain.mlg.jForest.sampler.SubsetSampler;
import be.uclouvain.mlg.jForest.splitting.Splitter;
import be.uclouvain.mlg.jForest.tree.Tree;
import be.uclouvain.mlg.jForest.tree.TreeIF;

/**
 * Generic definition of a random tree ensemble
 */
public class Forest {

	private Data d;
	private RandomSampler bootstrapSampler, mtrySampler;
	private Class<? extends Splitter> nodeSplitterClass;
	private final int mtry, maxDepth;
	private InternalImportanceIF inImp;
	private Tree[] trees;
	
	/**
	 * Grow a forest with the corresponding specifications
	 * @param d                    the training data
	 * @param ntree                the number of trees
	 * @param bootstrapSampler     the sampler that randomly selects training instances for growing each tree
	 * @param mtrySampler          the sampler that randomly selects the candidate variables for splitting
	 * @param nodeSplitterClass    the class of the objects that perform variable based splits
	 * @param inImp                the aggregator of variable importance computed during the tree growing process
	 * @param maxDepth             the maximal depth of the decision trees. A negative value will cause to fully grow the trees.
	 */
	public Forest(Data d, int ntree, RandomSampler bootstrapSampler, SubsetSampler mtrySampler, Class<? extends Splitter> nodeSplitterClass, InternalImportanceIF inImp, int maxDepth) {
		this.d = d;
		this.trees = new Tree[ntree];
		this.bootstrapSampler = bootstrapSampler;
		this.mtrySampler = mtrySampler;
		this.mtry = mtrySampler.getMtry();
		this.nodeSplitterClass = nodeSplitterClass;
		this.inImp = inImp;
		this.maxDepth = maxDepth;
		
		grow();
	}
	
	/**
	 * Grows the forest
	 */
	private void grow(){
		
		if(d.getNumberOfClasses() <= 1){
			RuntimeException up = new RuntimeException("There is only "+d.getNumberOfClasses()+" different labels in the dataset");
			throw up; // :-)
		}
		
		for(int i = 0; i < trees.length; i++){
			int[][] sampling = getSampling();
			
			trees[i] = new Tree(mtrySampler,nodeSplitterClass,d,sampling[0],sampling[1],inImp,new VariableCount(d.getP()),maxDepth);
			trees[i].grow();
		}
	}

	/**
	 * Ensures that at least two classes are present in the sampled instances
	 * @return an array of 2 elements :
	 *             <ol>
	 *                 <li>an array containing the indices of points in the sample</li>
	 *                 <li>an array containing the indices of points in the OOB</li>
	 *             </ol>
	 */
	private int[][] getSampling() {
		int[][] sampling;
		do{
			sampling = bootstrapSampler.getSample(d.getRowRange());
		}while(d.getNumberOfClassesIn(sampling[0])<=1); // we want at least 2 classes in a bootstrap sample
		return sampling;
	}
	
	/**
	 * Predicts the labels of new, previously unseen, samples
	 * @param newdata  the samples whose labels are to be predicted
	 * @return         a vector of class labels in the same order as in <code>newdata</code>
	 */
	public int[] predict(Data newdata){
		
		if(d.getP() != newdata.getP()){
			RuntimeException up = new RuntimeException("The number of cols of newdata ("+newdata.getP()+") does not correspond to the one of training data ("+d.getP()+")");
			throw up;
		}
		
		int[] res = new int[newdata.getN()];
		
		for(int i = 0; i < newdata.getN(); i++){ // for every new sample
			int[] classVote = new int[d.getNumberOfClasses()];
			for(TreeIF t : trees){
				classVote[t.predict(newdata.getData()[i])]++;
			}
			int bestScore = -1;
			for(int curClass = 0; curClass < classVote.length; curClass++){ // pr chaque classe 
				if(classVote[curClass] > bestScore){
					bestScore = classVote[curClass];
					res[i] = curClass;
				}
			}
		}
		
		return res;
		
	}
	
	
	/**
	 * Returns the importance of each variable as an aggregation of their importance in each split
	 * @return variable importance vector
	 */
	public double[] getInternalImportance(){
		return inImp.getImportances();
	}
	
	/**
	 * Computes variable importance after the forest is grown.
	 * This sequentially updates <code>eImp</code> for each tree.
	 * @param eImp an aggregator of variable importance
	 * @return a data structure containing the importance of the variables
	 */
	public <E> E getExternalImportance(ExternalImportanceIF<E> eImp){
		for(int i = 0; i < trees.length; i++){
			eImp.addImportanceOfTree(trees[i],i);
		}
		return eImp.getImportances();
	}
	
	/**
	 * @return the number of trees in the forest
	 */
	public int getNTree(){
		return trees.length;
	}
	
	/**
	 * @return the average tree depth 
	 */
	public double getAvgDepth(){
		double tmp = 0;
		for(TreeIF t : trees){
			tmp += t.getDepth();
		}
		return tmp / getNTree();
	}
	
	@Override
	public String toString(){
		return toString(false);
	}
	
	/**
	 * Sumarises the main parameters of the forest as a string
	 * @param verbose a boolean indicating if a string representation of all trees is to be computed
	 * @return a string representation of the forest
	 */
	public String toString(boolean verbose){
		
		String res = "Forest of "+trees.length+" trees, built on "+d.getN()+" samples with mtry = "+mtry+" and maxDepth = "+maxDepth;
		res += "\n\t avg depth :" + getAvgDepth();
		
		if(verbose){
			res += "\n\t internal feature importance : "+inImp;
			res += "\ndataset :\n";
			res+= d+"\n";
			
			for(int i = 0; i < trees.length; i++){
				res += "################################################ Tree "+i+" ################################################"+"\n";
				res += trees[i]+"\n";
			}
		}
		return res;
	}
}
