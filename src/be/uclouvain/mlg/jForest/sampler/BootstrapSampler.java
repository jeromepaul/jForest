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
 * Generates bootstrap samples from an original set <i>S</i> of indices.
 * A random sample consists of <i>|S|</i> indices sampled with replacement.
 * Each individual index has a .632 chance to be selected. 
 * 
 */
public class BootstrapSampler extends RandomSampler {

    /**
     * Creates an instance of BootstrapSampler
     * @param rnd a random number generator
     */
	public BootstrapSampler(Random rnd) {
		super(true, rnd);
	}

	@Override
	protected int[][] getSampleWithReplacement(int[] indices) {
		int[] sampling = new int[indices.length];
		boolean[] isPicked = new boolean[indices.length];
		int sizeOob = indices.length;
		
		for(int i = 0; i < indices.length; i++){
			int tmp = rnd.nextInt(indices.length);
			sampling[i] = indices[tmp];
			if(isPicked[tmp]) sizeOob++;
			isPicked[tmp] = true;
			sizeOob--;
		}
		
		int[] oob = new int[sizeOob];
		int c = 0;
		for(int i = 0; i < isPicked.length; i++){
			if(!isPicked[i]){
				oob[c] = indices[i];
				c++;
			}
		}
		
		int[][] res = new int[2][];
		res[0] = sampling;
		res[1] = oob;
		return res;
	}

	@Override
	protected int[][] getSampleWithoutReplacement(int[] indices) {
		RuntimeException up = new RuntimeException("sampling without replacement not implemented");
		throw up; // :-)
	}
	
}
