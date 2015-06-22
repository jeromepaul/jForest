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


#' Wrapper to jForest java framework
#' 
#' @name jForest
#' @docType package
#' @title jForest' R interface
#' @author Jérôme Paul, ICTEAM/Université catholique de Louvain
#'
#' @importFrom rJava .jarray
#' @importFrom rJava .jcall
#' @importFrom rJava .jcast
#' @importFrom rJava .jlong
#' @importFrom rJava .jnew
#' @importFrom rJava .jpackage
NULL


.onLoad <- function(libname, pkgname){
    .jpackage(pkgname,lib.loc=libname,jars=c('jForest.jar','commons-math3-3.3.jar'))
}

#' Transforms a dataframe with continuous and categorical attributes to a matrix.
#' The levels of a categorical variable are mapped to numeric values
#' @param d A data.frame
#' @return a matrix of numerics
formatData <- function(d){
    nMatrix = d
    
    for(i in labels(nMatrix)[[2]]){
        if(is.factor(nMatrix[[i]])){
            tmp = nMatrix[[i]]
            
            nMatrix[[i]] = suppressWarnings(as.numeric(as.vector(nMatrix[[i]])))
            if(any(is.na(nMatrix[[i]]))){ # names to numbers
                map = 1:nlevels(tmp)
                names(map) = levels(tmp)
                nMatrix[[i]] = as.numeric(map[tmp])
            }
        }
    }
    nMatrix = data.matrix(nMatrix);
    
    return(nMatrix);
}

#' Creates a random number generator for Java
#' @param seed an integer or \code{NULL}
#' @return an object of class java.util.Random initialized with \code{seed}
#' @export
rndFromSeed <- function(seed){
    if(is.null(seed)){
        .jnew("java/util/Random")
    }else{
        .jnew("java/util/Random",.jlong(seed))
    }
}

#' Creates an instance sampler that creates bootstrap samples of the data
#' @param seed an integer
#' @return an object of class sampler.RandomSampler initialized with \code{seed}
#' @export
bootstrapSampler <- function(seed){
    rnd = rndFromSeed(seed)
    o = .jnew("be/uclouvain/mlg/jForest/sampler/BootstrapSampler",rnd)
    .jcast(o,"be/uclouvain/mlg/jForest/sampler/RandomSampler")
}

#' Creates an instance sampler that always returns the full set of data
#' @return an object of class sampler.RandomSampler
#' @export
fullSetSampler <- function(){
    o = .jnew("be/uclouvain/mlg/jForest/sampler/FullSetSampler")
    .jcast(o,"be/uclouvain/mlg/jForest/sampler/RandomSampler")
}

#' Creates an instance sampler that creates sub-samples of the data
#' @param seed an integer
#' @param mtry the number of instances to be sampled
#' @return an object of class sampler.SubsetSampler initialized with \code{seed}
#' @export
subsetSampler <- function(seed, mtry){
    rnd = rndFromSeed(seed)
    .jnew("be/uclouvain/mlg/jForest/sampler/SubsetSampler",rnd,as.integer(mtry))
}

#' Creates a splitter that follows the CART method
#' @return an object of class splitting.CARTSplitter
#' @export
CARTSplitter <- function(){
    .jnew("be/uclouvain/mlg/jForest/splitting/CARTSplitter")$getClass()
}

#' Creates a splitter that follows the RRF method.
#' The Gini drop of features which are not favoured is multiplied by \code{coef}.
#' @param coef a number between 0 and 1.
#' @param favoured.features a logical vector which indicates for each feature 
#'                          if it is favoured or not. The features must be in the
#'                          same order than in the \code{x} data.frame given 
#'                          to \code{jForest}.
#' @return an object of class splitting.PriorKnowledgeCARTSplitter
#' @references Deng, Houtao, and George Runger.
#'             Feature selection via regularized trees.
#'             Neural Networks (IJCNN), The 2012 International Joint Conference on. IEEE, 2012.
#' @export
CARTwithPriorSplitter <- function(coef,favoured.features){
    cl = .jnew("be/uclouvain/mlg/jForest/splitting/PriorKnowledgeCARTSplitter")
    cl$setCoef(as.numeric(coef))
    cl$setFeatureToFavor(favoured.features)
    cl$getClass()
}

#' Creates a splitter that follows the Extra-trees method
#' i.e. choose one random split per feature and keep the best
#' @return an object of class splitting.ExtraTreesSplitter
#' @references Geurts, P., Ernst, D., & Wehenkel, L. (2006).
#'             Extremely randomized trees. Machine Learning, 63(1), 3-42.
#' @export
extraTreesSplitter <- function(){
    .jnew("be/uclouvain/mlg/jForest/splitting/ExtraTreesSplitter")$getClass()
}

#' Creates an aggregator of feature importance measures
#' @param p the number of features in the dataset
#' @return an object of class importance.internal.InternalImportanceIF
#' @export
averageSplitImportance <- function(p){
    o = .jnew("be/uclouvain/mlg/jForest/importance/internal/AverageSplitIndex",as.integer(p))
    .jcast(o,"be/uclouvain/mlg/jForest/importance/internal/InternalImportanceIF")
}

#' Builds a jForest classification model.
#' The default parameters corresponds to Breiman's Random Forest.
#' @param x a n*p data.frame containing n samples in p dimensions
#' @param y a vector of factors containing the n class labels
#' @param ntree the number of trees to be grown
#' @param mtry the number of candidate variables to be sampled in each node
#' @param seed an integer to initialize the randomization
#' @param instanceSampler an object of class sampler.RandomSampler
#' @param featureSampler an object of class sampler.SubsetSampler
#' @param splitCriterion an object of class splitting.CARTSplitter
#' @param inImportance an object of class importance.internal.InternalImportanceIF
#' @param maxDepth an integer defining the maximal depth of the trees.
#'                 If it is set to a negative value, trees are fully grown.
#' @return a jForest predictive model
#' @examples
#' m = jForest(iris[,1:4],iris$Species,ntree=100,seed=42)
#'
#' @export
jForest <- function(x,
                    y,
                    ntree=500,
                    mtry=sqrt(ncol(x)),
                    seed=NULL,
                    instanceSampler=bootstrapSampler(seed),
                    featureSampler=subsetSampler(if(is.null(seed)) seed else seed + 1,mtry),
                    splitCriterion=CARTSplitter(),
                    inImportance=averageSplitImportance(ncol(x)),
                    maxDepth=-1L){
    
    if(!is.data.frame(x)) stop("x must be a data.frame")
    
    x.matrix = .jarray(formatData(x)+.0,dispatch=TRUE)
    
    labels.map = levels(y)
    names(labels.map) = 0:(length(labels.map)-1)
    l = .jarray(as.integer(y)-1L)
    
    is.cat = .jarray(sapply(x,is.factor))
    
    d = .jnew("be/uclouvain/mlg/jForest/data/Data",x.matrix,l,is.cat)
    
    forest = .jnew("be/uclouvain/mlg/jForest/forest/Forest",
              d,
              as.integer(ntree),
              instanceSampler,
              featureSampler,
              splitCriterion,
              inImportance,
              as.integer(maxDepth))
    
    m = list(forest=forest,
             labels.map=labels.map,
             is.cat=is.cat,
             feat.names=colnames(x),
             data=d,
             samplingRnd=instanceSampler$getRandom())
    class(m) = "jForest"
    
    m
}

#' Classifies new data samples
#' @param object a jForest object
#' @param newdata a data.frame of new data
#' @param ... ignored
#' @return the class labels corresponding to the samples in \code{newdata}
#' @examples
#' m = jForest(iris[,1:4],iris$Species,ntree=100,seed=42)
#' predict(m,iris[,1:4])
#' 
#' @export
predict.jForest <- function(object,newdata,...){
    
    dummy.labels = .jarray(rep(-1L,nrow(newdata)))
    
    newdata = formatData(newdata)+.0
    newdata = .jarray(newdata,dispatch=TRUE)
    
    d = .jnew("be/uclouvain/mlg/jForest/data/Data",newdata,dummy.labels,object$is.cat)
    
    rawPred = .jcall(object$forest,"[I","predict",d)
    
    pred = object$labels.map[as.character(rawPred)]
    pred = factor(pred,levels=object$labels.map)
    names(pred) = NULL
    
    pred
}

#' Computes the importances of the variables in a jForest model
#' @param model a jForest model
#' @param type takes values in \code{c("internal","Ja","Jp","Jchisq","Jks","Jks.bcr")}.
#'             \code{"internal"} computes the importance specified by \code{inImportance} in \code{jForest} function.
#'             \code{"Ja"} computes Breiman's mean decrease in accuracy feature importance.
#'             \code{"Jp"} computes the mean difference in class prediction.
#'             \code{"Jchisq"} computes the significance of the features through a chi-squared test on the class vote distributions.
#'             \code{"Jks"} computes the significance of the features through a Kolmogorov-Smirnov test on the accuracies.
#'             \code{"Jks.bcr"} computes the significance of the features through a Kolmogorov-Smirnov test on the BCRs.
#' @param pval a boolean indicating if \code{"Jchisq"}, \code{"Jks"} or \code{"Jks.bcr"} computes p-values or the value of the chi-squared statistic.
#' @param fdr a boolean indicating if the p-values of \code{"Jchisq"}, \code{"Jks"} or \code{"Jks.bcr"} must be corrected for multiple testing with the FDR correction (see \code{? p.adjust}).
#' @return a vector containing the importance of each variables.
#' @references Jerome Paul, Pierre Dupont,
#'             Inferring statistically significant features from random forests,
#'             Neurocomputing, Volume 150, Part B, 20 February 2015, Pages 471-480, ISSN 0925-2312,
#'             \url{http://dx.doi.org/10.1016/j.neucom.2014.07.067}.
#' @examples
#' m = jForest(iris[,1:4],iris$Species,ntree=1000,seed=42)
#' predict(m,iris[,1:4])
#' importance(m)
#' importance(m,"Ja")
#' importance(m,"Jchisq")
#' importance(m,"Jks")
#' 
#' @export
importance <- function(model,type="internal",pval=TRUE,fdr=TRUE){
    
    feat.imp = NULL
    
    if(type == "internal"){
        feat.imp = model$forest$getInternalImportance()
    }else if(type == "Ja"){
        acc.drop = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/indices/AccuracyDrop")
        acc.drop = .jcast(acc.drop,"be/uclouvain/mlg/jForest/importance/external/permutation/indices/PermutationIndexIF")
        
        average = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/Average",model$data,model$samplingRnd,acc.drop)
        
        average = .jcast(average,"be/uclouvain/mlg/jForest/importance/external/ExternalImportanceIF")
        
        feat.imp = model$forest$getExternalImportance(average)
        
    }else if(type == "Jp"){
        diff.pred = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/indices/DifferenceInPrediction")
        diff.pred = .jcast(diff.pred,"be/uclouvain/mlg/jForest/importance/external/permutation/indices/PermutationIndexIF")
        
        average = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/Average",model$data,model$samplingRnd,diff.pred)
        
        average = .jcast(average,"be/uclouvain/mlg/jForest/importance/external/ExternalImportanceIF")
        
        feat.imp = model$forest$getExternalImportance(average)
    }else if(type == "Jchisq"){
        cont.table = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/indices/TwoConfusionMatrices",model$data$getNumberOfClasses())
        cont.table = .jcast(cont.table,"be/uclouvain/mlg/jForest/importance/external/permutation/indices/PermutationIndexWithDimIF")
        
        chisq.test = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/ChisqOnTwoConfusionTables",model$data,model$samplingRnd,cont.table,pval)
        
        feat.imp = model$forest$getExternalImportance(chisq.test)
    }else if(type == "Jks"){
        accAndPermAcc = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/indices/AccAndPermAcc")
        accAndPermAcc = .jcast(accAndPermAcc,"be/uclouvain/mlg/jForest/importance/external/permutation/indices/PermutationIndexIF")
        
        ks = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/KSTestOnTreePredPerfs",model$data,model$samplingRnd,accAndPermAcc,model$forest$getNTree(),pval)
        
        feat.imp = model$forest$getExternalImportance(ks)
    }else if(type == "Jks.bcr"){
        bcrAndPermBcr = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/indices/BcrAndPermBcr",length(model$labels.map))
        bcrAndPermBcr = .jcast(bcrAndPermBcr,"be/uclouvain/mlg/jForest/importance/external/permutation/indices/PermutationIndexIF")
        
        ks = .jnew("be/uclouvain/mlg/jForest/importance/external/permutation/KSTestOnTreePredPerfs",model$data,model$samplingRnd,bcrAndPermBcr,model$forest$getNTree(),pval)
        
        feat.imp = model$forest$getExternalImportance(ks)
    }
    
    if(!is.null(feat.imp)){
        if(type %in% c("Jchisq","Jks","Jks.bcr") && pval && fdr){
            feat.imp = p.adjust(feat.imp,"fdr")
        }
        
        names(feat.imp) = model$feat.names
        
    }else{
        stop("no importance computed")
    }
    
    feat.imp
}

