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

package be.uclouvain.mlg.jForest.sampler;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Generates bootstrap samples from an original set <i>S</i> of indices.
 * A random sample consists of <i>mtry</i> indices sampled without replacement.
 * There are <i>C(|S|,mtry)</i> different sets of size <i>mtry</i> in <i>S</i>. 
 */
public class SubsetSampler extends RandomSampler {

	protected final int mtry;

	/**
	 * Creates an instance of SubsetSampler
	 * @param rnd a random number generator
	 * @param mtry the number of indices to be sampled. <i>mtry < |S| </i> should hold.
	 */
	public SubsetSampler(Random rnd, int mtry) {
		super(false, rnd);
		this.mtry = mtry;
	}
	

	@Override
	protected int[][] getSampleWithReplacement(int[] indices) {
		RuntimeException up = new RuntimeException("Sampling with replacement not implemented");
		throw up;
	}

	@Override
	protected int[][] getSampleWithoutReplacement(int[] indices) {
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		for(int i : indices) tmp.add(i);
		Collections.shuffle(tmp,rnd);
		int[] in = new int[mtry];
		int[] oob = new int[indices.length-mtry];
		for(int i = 0; i < mtry; i++){
			in[i] = tmp.get(i);
		}
		for(int i = 0; i < indices.length-mtry; i++){
			oob[i] = tmp.get(mtry+i);
		}

		int[][] res = new int[2][];
		res[0] = in;
		res[1] = oob;
		return res;
	}
	
	/**
	 * @return the size of the subsets to be sampled
	 */
	public int getMtry(){
		return mtry;
	}

}
