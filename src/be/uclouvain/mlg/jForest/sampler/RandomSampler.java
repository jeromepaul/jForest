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
import java.util.Random;

/**
 * Generic definition of a random sampler
 */
public abstract class RandomSampler {
	
	protected Random rnd;
	protected boolean replacement;
	
	/**
	 * Creates an instance of RandomSampler
	 * @param replacement a boolean indicating if the sampling is to be done with replacement
	 * @param rnd a random number generator
	 */
	public RandomSampler(boolean replacement,Random rnd){
		this.replacement = replacement;
		this.rnd = rnd;
	}
	
	/**
	 * Get a sample of the array of indices
	 * @param indices is an array containing the indices of available points in the dataset
	 * @pre the current object is initialised
     * @return an array of 2 elements :
     *             <ol>
     *                 <li>an array containing the indices of points in the sample</li>
     *                 <li>an array containing the indices of points in the OOB</li>
     *             </ol>
	 */
	public int[][] getSample(int[] indices){
		if(replacement){
			return getSampleWithReplacement(indices);
		}
		else{
			return getSampleWithoutReplacement(indices);
		}
	}
	
	/**
	 * @return the random number generator used to perform the sampling
	 */
	public Random getRandom(){
		return rnd;
	}
	
	/**
	 * Implements sampling with replacement <i>cf.</i> <code>public int[][] getSample(int[] indices)</code>
	 */
	protected abstract int[][] getSampleWithReplacement(int[] indices);
	
	/**
	 * Implements sampling without replacement <i>cf.</i> <code>public int[][] getSample(int[] indices)</code>
	 */
	protected abstract int[][] getSampleWithoutReplacement(int[] indices);
	
}
