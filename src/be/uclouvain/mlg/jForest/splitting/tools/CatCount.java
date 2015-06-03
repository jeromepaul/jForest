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

package be.uclouvain.mlg.jForest.splitting.tools;

/**
 * This class defines helper objects that compute class counts/probabilities
 * for a particular level of a categorical variable.
 * (see computational shortcut in CARTSplitter)
 */
public class CatCount implements Comparable<CatCount> {

	private double level;
	private int n0 = 0, n1 = 0, c0, c1;
	
	/**
	 * @param level the level of interest of the categorical variable 
	 * @param c0 the label of one class
	 * @param c1 the label of the other class
	 * @pre   the classification problem should involve only two classes
	 */
	public CatCount(double level, int c0, int c1){
		this.c0 = c0;
		this.c1 = c1;
		this.level = level;
	}
	
	/**
	 * Increments the count of one class.
	 * Each time the variables takes value <code>level</code> this methods should be called
	 * to update the class probabilities.
	 * @param label either <code>c0</code> or <code>c1</code> 
	 */
	public void add(int label){
		if(label == c0) n0++;
		else n1++;
	}

	/**
	 * Computes the class probability of <code>c0</code> based on the observed classes.
	 * @return the class probability of <code>c0</code> or 0 if <code>add(int label)</code> has not been called.
	 */
	public double getProb0(){
		if(n0 + n1 == 0) return 0;
		return ((double) n0) / (n0 + n1);
	}
	
	/**
	 * @return the level of interest of the categorical variable
	 */
	public double getLevel(){
		return level;
	}
	
	/**
	 * @return the label of one class
	 */
	public int getC0(){
		return c0;
	}
	
	/**
	 * @return the label of the other class
	 */
	public int getC1(){
		return c1;
	}
	
	/**
	 * @return the number of occurence of class <code>c0</code>
	 */
	public int getN0(){
		return n0;
	}
	
	/**
	 * @return the number of occurence of class <code>c1</code>
	 */
	public int getN1(){
		return n1;
	}
	
	@Override
	public int compareTo(CatCount o) {
		return Double.compare(this.getProb0(), o.getProb0());
	}
	
	@Override
	public String toString() {
		return "cat "+level+": class0 "+c0+": (n0, n1, prob0) = ("+n0+", "+n1+", "+getProb0()+")";
	}
	
}