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
    
    parser.add_argument(
        "--inFileNames",
        help = "Input file names (file1:tree1 file2:tree2 ...)",
        type = str,
        nargs = "*",
        required = True,
    )
    
    parser.add_argument(
        "--outFileName",
        help = "Output file name",
        type = str,
        required = True,
    )
    
    
    args = parser.parse_args()
    d_args = vars(args)
    
    
    d_config = utils.load_config(args.config)
    d_modelConfig = utils.load_config(d_config["modelConfig"])
    #print(d_config)
    
    
    imgSpec = ImageSpec(
        nBinX = d_modelConfig["xVar"]["nBin"],
        nBinY = d_modelConfig["yVar"]["nBin"],
        
        xMin = d_modelConfig["xVar"]["min"],
        xMax = d_modelConfig["xVar"]["max"],
        
        yMin = d_modelConfig["yVar"]["min"],
        yMax = d_modelConfig["yVar"]["max"],
    )
    
    xVar = "min(max(0, ({xVar} - {xMin}) / {xBinWidth}), {nBinX})".format(
        xVar = d_modelConfig["xVar"]["branch"],
        xMin = imgSpec.xMin,
        xBinWidth = imgSpec.xBinWidth,
        nBinX = imgSpec.nBinX-1,
    )
    
    yVar = "min(max(0, ({yVar} - {yMin}) / {yBinWidth}), {nBinY})".format(
        yVar = d_modelConfig["yVar"]["branch"],
        yMin = imgSpec.yMin,
        yBinWidth = imgSpec.yBinWidth,
        nBinY = imgSpec.nBinY-1,
    )
    
    #brName_nConsti = d_modelConfig["constiBranch"]
    #nEventPerJob = d_modelConfig["loadEventPerJob"]
    
    l_layerInfo = []
    
    for iLayer, layer in enumerate(d_modelConfig["layers"]) :
        
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
    
    for modelKey in d_config["modelFiles"].keys() :
        
        modelFile = "%s/%s/%s" %(d_config["modelDir"], d_config["modelName"], d_config["modelFiles"][modelKey])
        
        d_classifierBranchName = {}
        
        for classifierKey in d_config["classifiers"] :
            
            classifierName = "%s_%s" %(modelKey, classifierKey)
            
            d_classifierBranchName[classifierName] = d_config["classifiers"][classifierKey]
        
        d_model[modelKey] = {
            "model": tensorflow.keras.models.load_model(modelFile, compile = False),
            "classifiers": d_classifierBranchName,
        }
    
    print(d_model)
    
    fileAndTreeNames = utils.get_fileAndTreeNames(args.inFileNames)
    nFile = len(fileAndTreeNames)
    
    if ("/" in args.outFileName) :
        
        outDir = args.outFileName[0: args.outFileName.rfind("/")]
        os.system("mkdir -p %s" %(outDir))
    
    outFileName = "file:%s" %(args.outFileName) if (args.outFileName.find("file:") != 0) else args.outFileName
    outFile = ROOT.TFile.Open(outFileName, "RECREATE")
    outTree = ROOT.TTree("tree", "tree")
    
    d_classifierBranch = {}
    
    for modelKey in d_model.keys() :
        
        for classifierKey in d_model[modelKey]["classifiers"].keys() :
            
            d_classifierBranch[classifierKey] = ROOT.std.vector("double")()
            outTree.Branch(classifierKey, d_classifierBranch[classifierKey])
    
    d_ypred = {}
    
    for iFile, fileAndTreeName in enumerate(fileAndTreeNames) :
        
        verbosetag = "[file %d/%d]" %(iFile+1, nFile)
        
        print("%s input : %s " %(verbosetag, fileAndTreeName))
        print("%s output: %s " %(verbosetag, outFileName))
        #exit()
        
        nEvent_processed = 0
        
        for branches in uproot.iterate(
            files = fileAndTreeName,
            expressions = d_branchName_alias.keys(),
            aliases = d_branchName_alias,
            ##cut = args.cut,
            language = utils.uproot_lang,
            step_size = 10000,
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
                
                
                if (nEvent_processed == 1 or not (nEvent_processed % d_config["printEvery"])) :
                    
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
    
    
    print("Processing successful. Saving output tree...")
    
    outFile.cd()
    outTree.Write()
    outFile.Close()
    
    print("Output tree successfully saved.")
    
    return 0



if (__name__ == "__main__") :
    
    main()
