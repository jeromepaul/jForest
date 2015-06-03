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

import java.util.Random;

import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import be.uclouvain.mlg.jForest.data.Data;
import be.uclouvain.mlg.jForest.importance.external.permutation.indices.PermutationIndexIF;

/**
 * Given two predictive performances for each tree - one for the original OOB and one with a permuted OOB - computes a KS test for each variable.
 * See J. Paul and P. Dupont,
 * <a href="http://sites.uclouvain.be/benelearn2014/wp-content/uploads/2014/06/proceedings.pdf" target="_blank">Statistically interpretable importance indices for Random Forests</a>,
 * 23rd Annual Machine Learning Conference of Belgium and the Netherlands (BENELEARN), p. 7, Brussels, Belgium, June 6, 2014.
 * This is explained in <a href="http://jeromepaul.be/wp-content/uploads/2014/06/2014BENELEARN_slides.pdf" target="_blank">the slides</a> of the talk given at BENELEARN.
 */
public class KSTestOnTreePredPerfs extends PermImportance<double[],double[]> {

	private int nfeat;
	private boolean returnPval;
	private double[] perfs; // perfs[t] = ACC or BCR or ... of tree t on it's OOB
	private double[][] permPerfs; // perfs[f][t] = ACC or BCR or ... of tree t on it's OOB when feat f is permuted
	
	/**
	 * Initialize an object that aggregates original and permuted predictive performances with a KS test
     * @param d the dataset
     * @param rnd a random number generator used to create permutations of the OOB
     * @param permutationIndex an object that computes a statistic for one variable in a particular tree
	 * @param ntree the number of trees in the forest
	 */
	public KSTestOnTreePredPerfs(Data d, Random rnd, PermutationIndexIF<double[]> permutationIndex, int ntree, boolean pval) {
		super(d, rnd, permutationIndex);
		nfeat = d.getP();
		perfs = new double[ntree];
		permPerfs = new double[nfeat][ntree];
		this.returnPval = pval;
	}

	@Override
	public double[] getImportances() {
		KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
		
		double[] res = new double[nfeat];
		for(int i = 0; i < nfeat; i++){
		    if(returnPval){
		        res[i] = ks.kolmogorovSmirnovTest(perfs, permPerfs[i]); 
		    }
		    else{
		        res[i] = ks.kolmogorovSmirnovStatistic(perfs, permPerfs[i]);
		    }
		}
		
		return res;
	}

	@Override
	protected void addStatOfVar(int v, double[] index, int treeId) {
		// index[0] = acc, index[1] = permAcc
		perfs[treeId] = index[0];
		permPerfs[v][treeId] = index[1];
	}

}
