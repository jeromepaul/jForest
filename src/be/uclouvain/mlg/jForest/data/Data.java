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

package be.uclouvain.mlg.jForest.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.TreeSet;

/**
 * Non sparse representation of a data matrix with class labels.
 * The variables can be either continuous or categorical.
 */
public class Data {
	private double[][] d;
	private int[] labels;
	private boolean[] isCat;
	private int n, p;
	private int nLabelLvl;
	private int[] labelLvls;
	
	private int[] rowRange;
	private int[] colRange;
	
	private double[][] catLevels;
	
	/**
	 * Creates a Data object
	 * @param d        the data matrix consisting of <code>n</code> samples in a <code>p</code> dimensional space.
	 *                 The levels of categorical variables should be encoded as numeric (per dimension).
	 *                 <code>d[i][j]</code> is the value of the <i>j</i>-th dimension of the <i>i</i>-th data point.
	 * @param labels   the <code>n</code> class labels encoded as integers from 0 to <i>number of classes</i> - 1
	 * @param isCat    a boolean vector of size <code>p</code> indicating for each variable if it is categorical (<code>true</code>) or continuous (<code>false</code>).
	 */
	public Data(double[][] d,int[] labels, boolean[] isCat){
		this.d = d;
		this.labels = labels;
		this.isCat = isCat;
		n = d.length;
		p = d[0].length;
		
		rowRange = new int[n];
		for(int i = 0; i < n; i++) rowRange[i] = i;
		nLabelLvl = getNumberOfClassesIn(rowRange);
		
		colRange = new int[p];
		for(int i = 0; i < p; i++) colRange[i] = i;
		
		if(n != labels.length) throw new RuntimeException("The size of the label vector is different from the number of samples in data.");
		if(p != isCat.length) throw new RuntimeException("The size of the isCat vector is different from the number of dimensions.");
		
		// for each categorical feature, store a table with one occurrence of each level
		catLevels = new double[p][];
		for(int i = 0; i < p; i++){ // i is the feature index
			if(isCat[i]){
				HashSet<Double> tmp = new HashSet<Double>();
				for(int s = 0; s < n; s++){ // s for sample
					tmp.add(d[s][i]);
				}
				double[] tmpLvl = new double[tmp.size()];
				int idTmp = 0;
				for(double cur : tmp){
					tmpLvl[idTmp] = cur;
					idTmp++;
				}
				catLevels[i] = tmpLvl;
			}
		}
		
		TreeSet<Integer> lablvls = new TreeSet<Integer>();
		for(int l : labels) lablvls.add(l);
		labelLvls = new int[lablvls.size()];
		for(int i = 0; i < labelLvls.length; i++) labelLvls[i] = lablvls.pollFirst();
		
	}
	
	/**
	 * The data is represented as an two dimensional array.
	 * Let <code> double[][] dataMatrix = d.getData();</code> then <code>d[i][j]</code> is the value of the <i>j</i>-th dimension of the <i>i</i>-th data point. 
	 * @return the data matrix
	 */
	public double[][] getData(){
		return d;
	}
	
	/**
	 * @return the number <code>n</code> of rows of the data matrix
	 */
	public int getN(){
		return n;
	}
	
	/**
	 * @return the number <code>p</code> of columns of the data matrix
	 */
	public int getP(){
		return p;
	}
	
	/**
	 * @return the label vector
	 */
	public int[] getLabels(){
		return labels;
	}
	
	/**
	 * @return one occurrence of each class label (sorted by increasing order) 
	 */
	public int[] getLabelLvls(){
		return labelLvls;
	}
	
	/**
	 * @return a vector indicating if variables are categorical
	 */
	public boolean[] getIsCat(){
		return isCat;
	}
	
	/**
	 * @return the number of different class labels
	 */
	public int getNumberOfClasses(){
		return nLabelLvl;
	}
	
	/**
	 * Computes the number of classes in a subset of the dataset.
	 * @param sampleIds is an array of samples indices to be taken into account
	 * @return the number of classes in the samples of samplesIds
	 */
	public int getNumberOfClassesIn(int[] sampleIds){
		HashSet<Integer> s = new HashSet<Integer>();
		for(int i : sampleIds){
			s.add(labels[i]);
		}
		return s.size();
	}
	
	/**
	 * For a categorical variable, gives an array of levels. For continuous variables, returns <code>null</code>.
	 * @param feat is the index of a variable
	 * @return an array containing one occurrence of each level of variable feat
	 */
	public double[] getLevelsOfCatVar(int feat){
		return catLevels[feat];
	}
	
	/**
	 * @return an array containing [0,1,...,nrow-1]
	 */
	public int[] getRowRange(){
		return rowRange;
	}
	
	/**
	 * @return an array containing [0,1,...,ncol-1]
	 */
	public int[] getColRange(){
		return colRange;
	}
	
	/**
	 * Computes the majority class in a subsample of the dataset 
	 * @param sampleIds    an array indicating the indices of the samples to consider
	 * @return             an array containing the majority classes.
	 *                     It contains more than one element only if several classes are equally most represented.
	 */
	public int[] getMajorityClass(int[] sampleIds){
		int[] classCount = new int[getNumberOfClasses()];
		for(int i : sampleIds) classCount[getLabels()[i]]++;
		int bestCount = -1;
		for(int i = 0; i < classCount.length; i++){
			if(classCount[i] > bestCount){
				bestCount = classCount[i];
			}
		}
		
		ArrayList<Integer> bestClasses = new ArrayList<Integer>(classCount.length);
		for(int i = 0; i < classCount.length; i++){
			if(classCount[i] == bestCount) bestClasses.add(i);
		}
		
		int[] maxClass = new int[bestClasses.size()];
		for(int i = 0; i < bestClasses.size(); i++) maxClass[i] = bestClasses.get(i);
		
		return maxClass;
	}
	
	@Override
	public String toString(){
		String res = "Data :\n";
		for(double[] i : d){
			res += Arrays.toString(i) + "\n";
		}
		res += "Labels :\n"+Arrays.toString(labels) + "\n";
		res += "isCat :\n"+Arrays.toString(isCat) + "\n";
		res += "nLabelLvl : "+nLabelLvl + "\n";
		res += "rowRange : "+Arrays.toString(rowRange) + "\n";
		res += "colRange : "+Arrays.toString(colRange) + "\n";
		res += "categorical vars + levels :\n";
		for(int i = 0; i < catLevels.length; i++){
			res += "\ti = "+i+" --> "+Arrays.toString(catLevels[i]);
		}
		return res;
	}
}
