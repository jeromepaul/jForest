# jForest
jForest is a general framework for Machine Learning. It implements tree ensemble based classification methods. It is designed to be very modular and allows easy tuning and modification of the tree induction, classification criterion and feature importance index.

## Installation
### from the R package
Download the [jForest package](https://github.com/jeromepaul/jForest/raw/master/jForest_1.0.tar.gz).
Then run the following line in a shell
```
R CMD INSTALL jForest_1.0.tar.gz
```
You only need the <a href="http://cran.r-project.org/web/packages/rJava/" target="_blank">rJava</a> library to be installed.

### from source
Clone this repository and run
```
./makeRPackage.sh
```
You will need the <a href="http://cran.r-project.org/web/packages/rJava/" target="_blank">rJava</a> and <a href="http://cran.r-project.org/web/packages/roxygen2/" target="_blank">roxygen2</a> libraries.

## Documentation
* The *javadoc* is available <a href="http://jeromepaul.github.io/jForest/javadoc/" target="_blank">here</a>.
* The documentation of the *R package* is available <a href="http://jeromepaul.github.io/jForest/R-manual/jForest-manual.pdf" target="_blank">here</a>.

## Copyright and License
Copyright © 2010-2015, Université catholique de Louvain, Belgium - UCL
All rights reserved.
 
jForest has been developed by <a href="http://jeromepaul.be/" target="_blank">Jérôme Paul</a> (<a href="http://uclouvain.be/mlg" target="_blank">Machine Learning Group (MLG)</a> - <a href="http://www.uclouvain.be/en-icteam.html" target="_blank">Institute of Information and Communication Technologies, Electronics and Applied Mathematics (ICTEAM)</a>) for the <a href="http://www.uclouvain.be" target="_blank">Université catholique de Louvain (UCL)</a>.

It is distributed under the <a href="http://www.gnu.org/licenses/gpl.html" target="_blank">GNU General Public License</a>.

