from __future__ import print_function

#import mxnet
import argparse
import awkward
import collections
#import cppyy
#import cppyy.ll
import concurrent.futures
import datetime
import gc
import keras
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import memory_profiler
import multiprocessing
import multiprocessing.managers
import multiprocessing.shared_memory
import numpy
import os
import PIL
#import pprint
import psutil
import pympler
import ROOT
import sklearn
import sklearn.metrics
import sortedcontainers
import sparse
import sys
import tabulate
import tensorflow
import tensorflow.keras
import time
import uproot
import yaml

#from tensorflow.keras import datasets, layers, models
#from tensorflow.keras import mixed_precision

#policy = mixed_precision.Policy("mixed_float16")
#mixed_precision.set_global_policy(policy)

import utils


class ImageSpec :
    
    def __init__(
        self,
        nBinX, nBinY,
        xMin, xMax,
        yMin, yMax,
    ) :
        
        self.nBinX = nBinX
        self.nBinY = nBinY
        
        self.xMin = numpy.float16(xMin)
        self.xMax = numpy.float16(xMax)
        
        self.yMin = numpy.float16(yMin)
        self.yMax = numpy.float16(yMax)
        
        self.xBinWidth = (self.xMax-self.xMin) / nBinX
        self.yBinWidth = (self.yMax-self.yMin) / nBinY


class LayerInfo :
    
    def __init__(
        self,
        layerIdx,
        imgSpec,
        valueBranch,
        cutBranch,
        resolverOperation,
        resolverUpdate = None,
        resolverBranch = None,
    ) :
        
        self.layerIdx = layerIdx
        
        self.imgSpec = imgSpec
        
        self.valueBranch = valueBranch
        self.cutBranch = cutBranch
        
        self.resolverOperation = resolverOperation
        self.resolverUpdate = resolverUpdate
        
        self.resolverBranch = resolverBranch
    
    
    def resolveEntry(
        self,
        oldValue,
        newValue,
        oldResolver,
        newResolver,
    ) :
        
        updatedValue = None
        updatedResolver = None
        
        updatedValue = eval(self.resolverOperation.format(
            oldValue = oldValue,
            newValue = newValue,
            oldResolver = oldResolver,
            newResolver = newResolver,
        ))
        
        
        if (self.resolverUpdate is not None) :
            
            updatedResolver = eval(self.resolverUpdate.format(
                oldValue = oldValue,
                newValue = newValue,
                oldResolver = oldResolver,
                newResolver = newResolver,
            ))
        
        
        return (updatedValue, updatedResolver)



def main() :
    
    # Argument parser
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "--config",
        help = "Configuration file",
        type = str,
        required = True,
    )
    
    
    args = parser.parse_args()
    d_args = vars(args)
    
    
    d_loadConfig = utils.load_config(args.config)
    #print(d_loadConfig)
    
    
    imgSpec = ImageSpec(
        nBinX = d_loadConfig["xVar"]["nBin"],
        nBinY = d_loadConfig["yVar"]["nBin"],
        
        xMin = d_loadConfig["xVar"]["min"],
        xMax = d_loadConfig["xVar"]["max"],
        
        yMin = d_loadConfig["yVar"]["min"],
        yMax = d_loadConfig["yVar"]["max"],
    )
    
    xVar = "min(max(0, ({xVar} - {xMin}) / {xBinWidth}), {nBinX})".format(
        xVar = d_loadConfig["xVar"]["branch"],
        xMin = imgSpec.xMin,
        xBinWidth = imgSpec.xBinWidth,
        nBinX = imgSpec.nBinX-1,
    )
    
    yVar = "min(max(0, ({yVar} - {yMin}) / {yBinWidth}), {nBinY})".format(
        yVar = d_loadConfig["yVar"]["branch"],
        yMin = imgSpec.yMin,
        yBinWidth = imgSpec.yBinWidth,
        nBinY = imgSpec.nBinY-1,
    )
    
    #brName_nConsti = d_loadConfig["constiBranch"]
    #nEventPerJob = d_loadConfig["loadEventPerJob"]
    
    l_layerInfo = []
    
    for iLayer, layer in enumerate(d_loadConfig["layers"]) :
        
        l_layerInfo.append(LayerInfo(
            layerIdx = iLayer,
            imgSpec = imgSpec,
            **layer,
        ))
    
    
    nLayer = len(l_layerInfo)
    img_shape = (imgSpec.nBinY, imgSpec.nBinX, nLayer)
    
    xVarName = "xVar"
    yVarName = "yVar"
    nConstiVarName = "nConsti"
    
    wVarName_layer = "wVar_layer{layer}"
    constiCutVarName_layer = "constiCutVar_layer{layer}"
    resolverVarName_layer = "resolverVar_layer{layer}"
    
    d_branchName_alias = {
        xVarName: xVar,
        yVarName: yVar,
        #nConstiVarName: brName_nConsti,
    }
    
    for iLayer, layerInfo in enumerate(l_layerInfo) :
        
        wVarName = wVarName_layer.format(layer = iLayer)
        constiCutVarName = constiCutVarName_layer.format(layer = iLayer)
        resolverVarName = resolverVarName_layer.format(layer = iLayer)
        
        d_branchName_alias[wVarName] = layerInfo.valueBranch
        
        if (layerInfo.cutBranch is not None) :
            
            d_branchName_alias[constiCutVarName] = layerInfo.cutBranch
        
        if (layerInfo.resolverBranch is not None) :
            
            d_branchName_alias[resolverVarName] = layerInfo.resolverBranch
    
    
    d_model = {}
    
    for modelKey in d_loadConfig["modelFiles"].keys() :
        
        modelFile = "%s/%s/%s" %(d_loadConfig["modelDir"], d_loadConfig["modelName"], d_loadConfig["modelFiles"][modelKey])
        
        d_classifierBranchName = {}
        
        for classifierKey in d_loadConfig["classifiers"] :
            
            classifierName = "%s_%s" %(modelKey, classifierKey)
            
            d_classifierBranchName[classifierName] = d_loadConfig["classifiers"][classifierKey]
        
        d_model[modelKey] = {
            "model": tensorflow.keras.models.load_model(modelFile, compile = False),
            "classifiers": d_classifierBranchName,
        }
    
    print(d_model)
    
    nSample = len(d_loadConfig["samples"])
    
    for iSample, sampleSource in enumerate(d_loadConfig["samples"]) :
        
        fileAndTreeNames = utils.get_fileAndTreeNames([sampleSource])
        nFile = len(fileAndTreeNames)
        
        sourceFile = sampleSource.strip().split(":")[0]
        sampleName = sourceFile.split("/")[-1].split(".txt")[0]
        
        outputTag = d_loadConfig["outputTag"].strip()
        outputTag = "_%s" if len(outputTag) else ""
        
        outDir = "%s/%s/%s%s" %(d_loadConfig["outputDirBase"], d_loadConfig["modelName"], sampleName, outputTag)
        os.system("mkdir -p %s" %(outDir))
        
        d_ypred = {}
        
        for iFile, fileAndTreeName in enumerate(fileAndTreeNames) :
            
            fileName = fileAndTreeName.strip().split(":")[0].split("/")[-1]
            
            outFileName = "file:%s/%s" %(outDir, fileName)
            outFile = ROOT.TFile.Open(outFileName, "RECREATE")
            outTree = ROOT.TTree("tree", "tree")
            
            verbosetag = "[sample %d/%d] [file %d/%d]" %(iSample+1, nSample, iFile+1, nFile)
            
            print("Input  %s: %s" %(verbosetag, fileAndTreeName))
            print("Output %s: %s" %(verbosetag, outFileName))
            #exit()
            
            d_classifierBranch = {}
            
            for modelKey in d_model.keys() :
                
                for classifierKey in d_model[modelKey]["classifiers"].keys() :
                    
                    d_classifierBranch[classifierKey] = ROOT.std.vector("double")()
                    outTree.Branch(classifierKey, d_classifierBranch[classifierKey])
            
            nEvent_processed = 0
            
            for branches in uproot.iterate(
                files = fileAndTreeName,
                expressions = d_branchName_alias.keys(),
                aliases = d_branchName_alias,
                ##cut = args.cut,
                language = utils.uproot_lang,
                step_size = 1000,
                #entry_stop = 10,
                num_workers = 10,
            ) :
                
                nEvent = len(branches[xVarName])
                #nEvent = 2
                
                for iEvent in range(0, nEvent) :
                    
                    nEvent_processed += 1
                    
                    # Clear branches
                    for classifierKey in d_classifierBranch.keys() :
                        
                        d_classifierBranch[classifierKey].clear()
                    
                    nJet = len(branches[xVarName][iEvent])
                    
                    nConsti = awkward.count(branches[xVarName][iEvent]) * nLayer
                    
                    arr_idx = numpy.zeros((nConsti, 4), dtype = numpy.uint32)
                    arr_val = numpy.zeros(nConsti, dtype = numpy.float16)
                    
                    entryCount = 0
                    
                    for iJet in range(0, nJet) :
                        
                        a_x  = branches[xVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                        a_y  = branches[yVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                        
                        for iLayer, layerInfo in enumerate(l_layerInfo) :
                            
                            wVarName = wVarName_layer.format(layer = iLayer)
                            constiCutVarName = constiCutVarName_layer.format(layer = iLayer)
                            resolverVarName = resolverVarName_layer.format(layer = iLayer)
                            
                            a_w  = branches[wVarName][iEvent][iJet].to_numpy()#.astype(dtype = numpy.float16)
                            
                            a_constiCut = None
                            
                            if (layerInfo.cutBranch is not None) :
                                
                                a_constiCut = branches[constiCutVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                            
                            else :
                                
                                a_constiCut = numpy.ones(len(a_x), dtype = numpy.uint32)
                            
                            a_resolver = None
                            
                            if (layerInfo.resolverBranch is not None) :
                                
                                a_resolver = branches[resolverVarName][iEvent][iJet]#.to_numpy()
                            
                            a_nonzero_idx = a_constiCut.nonzero()[0]
                            
                            d_arrayIdx = sortedcontainers.SortedDict()
                            d_arrayResolver = sortedcontainers.SortedDict()
                            
                            for iConsti in a_nonzero_idx :
                                
                                key = (
                                    numpy.uint32(iJet),
                                    a_y[iConsti],
                                    a_x[iConsti],
                                    numpy.uint32(layerInfo.layerIdx),
                                )
                                
                                val = a_w[iConsti]
                                
                                resolver = a_resolver[iConsti] if (a_resolver is not None) else None
                                
                                if (key in d_arrayIdx) :
                                    
                                    val, resolver = layerInfo.resolveEntry(
                                        oldValue = d_arrayIdx[key],
                                        newValue = val,
                                        oldResolver = d_arrayResolver[key],
                                        newResolver = resolver,
                                    )
                                
                                d_arrayIdx[key] = val
                                d_arrayResolver[key] = resolver
                            
                            for key in d_arrayIdx.keys() :
                                
                                arr_idx[entryCount] = key
                                arr_val[entryCount] = d_arrayIdx[key]
                                
                                entryCount += 1
                            
                            
                            #a_x  = None
                            #a_y  = None
                            #a_w  = None
                    
                    
                    if (nEvent_processed == 1 or not (nEvent_processed % d_loadConfig["printEvery"])) :
                        
                        print("%s: processed event %d." %(verbosetag, nEvent_processed))
                    
                    #gc.collect()
                    
                    
                    input_shape = (nJet,) + img_shape
                    
                    arr_nonzero_idx = arr_val.nonzero()[0]
                    
                    arr_inputdata = tensorflow.sparse.reorder(tensorflow.sparse.SparseTensor(
                        indices = arr_idx[arr_nonzero_idx],
                        values = arr_val[arr_nonzero_idx],
                        dense_shape = input_shape,
                    ))
                    
                    
                    for modelKey in d_model.keys() :
                        
                        y_pred = d_model[modelKey]["model"].predict_on_batch(arr_inputdata)
                        
                        if (not len(d_ypred)) :
                            
                            nNode = y_pred.shape[1]
                            
                            for iNode in range(0, nNode) :
                                
                                d_ypred["node%d" %(iNode)] = "y_pred[:, %d]" %(iNode)
                        
                        for classifierKey in d_model[modelKey]["classifiers"].keys() :
                            
                            eval_str = d_model[modelKey]["classifiers"][classifierKey].format(**d_ypred)
                            arr_classifier = eval(eval_str).astype(numpy.float)
                            
                            # Fill branches
                            for ele in arr_classifier :
                                
                                d_classifierBranch[classifierKey].push_back(ele)
                    
                    # Fill tree
                    outTree.Fill()
                    
                    #if (nEvent_processed >= 100) :
                    #    
                    #    break
                
                #if (nEvent_processed >= 100) :
                #    
                #    break
            
            
            outFile.cd()
            outTree.Write()
            outFile.Close()
            
            
            
            ##outFile.close()
    
    
    
    
    return 0



if (__name__ == "__main__") :
    
    main()
