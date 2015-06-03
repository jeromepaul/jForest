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

/**
 * Generic definition of a leaf of a decision tree
 */
public class Leaf implements TreeIF {
	
	private int classToPredict;
	private int[] inTree, oob;
	
	public Leaf(int[] inTree, int[] oob, int classToPredict){
		this.classToPredict = classToPredict;
		this.inTree = inTree;
		this.oob = oob;
	}
	
	@Override
	public void grow(){} // nothing to do

	@Override
	public int predict(double[] featVec) {
		return classToPredict;
	}

	@Override
	public String toString(){
		return "Leaf : samples = "+Arrays.toString(inTree)+", class = "+classToPredict+"\n\n";
	}

	@Override
	public int[] getOob() {
		return oob;
	}

	@Override
	public int getDepth() {
		return 0;
	}
	
}
