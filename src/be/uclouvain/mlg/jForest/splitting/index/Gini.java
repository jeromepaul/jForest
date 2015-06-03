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

package be.uclouvain.mlg.jForest.splitting.index;

/**
 * Dynamically computes Gini index of a node
 */
public class Gini {

    private int[] labCounts;
    private int tot;
    
    public Gini(int nLabels){
        labCounts = new int[nLabels];
    }
    
	/**
	 * Add labels in a node
	 * @param i is the label
	 * @param nTimes is the number of times label i is added
	 */
    public void addLabel(int i, int nTimes) {
        labCounts[i] += nTimes;
        tot += nTimes;
    }

	/**
	 * Add one occurence of label i
	 * @param i is the label to add
	 */
    public void addLabel(int i) {
        labCounts[i]++;
        tot++;
    }

	/**
	 * Remove labels in current node
	 * @param i is the label to remove
	 * @param nTimes is the number of times label i is removed
	 */
    public void removeLabel(int i, int nTimes) {
        labCounts[i] -= nTimes;
        tot -= nTimes;
    }

	/**
	 * Remove one occurence of label i
	 * @param i is the label to remove
	 */
    public void removeLabel(int i) {
        labCounts[i]--;
        tot--;
    }

	/**
	 * Computes the Gini index according to labels
	 * that appear in the current node
	 * @return the gini index
	 */
    public double getGini() {
        double res = 1;
        for(int i : labCounts) res -= Math.pow(((double) i)/tot,2);
        
        return res;
    }

	/**
	 * The number of points whose labels are taken into account
	 * in the current node
	 * @return the number of points in the node
	 */
    public int getNbPoints() {
        return tot;
    }
	
    @Override
    public String toString() {
        String s = "GeneralGini : [";
        for(int i = 0; i < labCounts.length; i++){
            s += "cl"+i+":"+labCounts[i]+", ";
        }
        s += "tot:"+getNbPoints()+"]";
        s += " gini = "+getGini();
        
        return s;
    }
    
	public static double getGiniDrop(Gini gCur, Gini gLeft, Gini gRight){
		if(gLeft.getNbPoints() + gRight.getNbPoints() != gCur.getNbPoints())
			throw new RuntimeException("|left "+gLeft.getNbPoints()+"|+|right "+gRight.getNbPoints()+"| != |cur "+gCur.getNbPoints()+"|");
		
		if(gLeft.getNbPoints() == 0 || gRight.getNbPoints() == 0) return 0; // There is no split... The data remains the same
		
		return gCur.getGini() - ((double) gLeft.getNbPoints())/gCur.getNbPoints()*gLeft.getGini() - ((double) gRight.getNbPoints())/gCur.getNbPoints()*gRight.getGini();
	}
	
}