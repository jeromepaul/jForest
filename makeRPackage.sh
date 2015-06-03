#! /bin/bash
# 
# jForest
# Copyright © 2010-2015, Université catholique de Louvain, Belgium - UCL
# All rights reserved.
# 
# This file is part of the jForest library.
# 
# jForest has been developed by Jérôme Paul
# (Machine Learning Group (MLG) - Institute of Information and Communication
# Technologies, Electronics and Applied Mathematics (ICTEAM)) for the
# Université catholique de Louvain (UCL). jForest is a general framework for
# Machine Learning. It implements tree ensemble based classification methods.
# It is designed to be very modular and allows easy tuning and modification of
# the tree induction, classification criterion and feature importance index.
# 
# jForest is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# 
# jForest is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with jForest.  If not, see <http://www.gnu.org/licenses/>.

rpackage="jForest"

# compile java files
if [ -a bin ]; then
    rm -r bin
fi

mkdir bin

for f in $(find src -name "*.java"); do
    echo "Compiling $f"
    javac -cp lib/commons-math3-3.3.jar:src -d bin $f
done

jar cf jForest.jar -C bin be

# make R package
echo "Creating R source package"

if [ -a $rpackage ]; then
    rm -r $rpackage
fi

mkdir $rpackage
cp DESCRIPTION $rpackage
cp LICENSE $rpackage

# copying binaries & exec
mkdir -p $rpackage/inst/java
cp jForest.jar $rpackage/inst/java
cp lib/commons-math3-3.3.jar $rpackage/inst/java

mkdir $rpackage/R
cp jForest.R $rpackage/R

# copying sources & doc
if [ -a doc ]; then
    rm -r doc
fi
javadoc $(find src -name "*.java") -encoding utf8 -classpath lib/commons-math3-3.3.jar -d doc -tag pre:a:"Preconditions:" -tag post:a:"Postconditions:"
jar cf jForest-src.jar src doc

mkdir $rpackage/java
cp jForest-src.jar $rpackage/java
cp lib/commons-math3-3.3-src.zip $rpackage/java

# create R doc and package
Rscript -e "setwd('$rpackage'); library(roxygen2); roxygenize()"
R CMD build $rpackage

# check package
pkg=$(ls jForest_*.tar.gz)
R CMD check $pkg

