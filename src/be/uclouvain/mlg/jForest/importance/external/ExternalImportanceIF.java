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

package be.uclouvain.mlg.jForest.importance.external;

import be.uclouvain.mlg.jForest.importance.ImportanceIF;
import be.uclouvain.mlg.jForest.tree.TreeIF;

/**
 * This interface must be implemented in order to define objects that will compute the importance
 * of variables after the forest is grown, from a criterion independent of the forest induction.
 * For instance, those criteria can be based on the prediction of individual trees on their out-of-bag samples.
 * <code>E</code> is the type of the computed variable importance measure, usually <code>double[]</code>.
 */
public interface ExternalImportanceIF<E> extends ImportanceIF<E> {
	
	/**
	 * Add the importance of variables in tree 'tree'
	 * @param tree is a tree of the forest
	 * @pre tree is initialised and grown
	 */
	public void addImportanceOfTree(TreeIF tree, int treeId);
}
