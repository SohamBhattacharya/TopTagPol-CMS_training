# https://sungwookyoo.github.io/tips/ArgParser/


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
import sklearn
import sklearn.metrics
import sortedcontainers
import sparse
import sys
import tabulate
import tensorflow
import time
import uproot
import yaml

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

import networks
import utils
import utils_tensorflow



# https://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_10_5_0/doc/html/dc/d55/classreco_1_1PFCandidate.html#ac55e8d12f21ad7c3fb6400f7fa60690c
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


class CategoryInfo :
    
    def __init__(
        self,
        catNum,
        catName,
        samples,
        l_layerInfo,
        cut,
    ) :
        
        self.catNum = catNum
        self.catName = catName
        self.l_sample = samples
        self.l_layerInfo = l_layerInfo
        self.cut = cut
        
        self.nJet = 0
        
        self.l_sample_fileAndTreeName = [None] * len(samples)
        self.l_sample_weight = [None] * len(samples)
        
        self.l_sample_nJet = [0] * len(samples)
        
        self.l_sample_nJetMax = [0] * len(samples)
        
        for iSample, sample in enumerate(samples) :
            
            l_token = sample.strip().split(":")
            
            fName = l_token[0]
            treeName = l_token[1]
            weight = float(l_token[2])
            nJetMax = int(l_token[3])
            
            self.l_sample_fileAndTreeName[iSample] = ["%s:%s" %(ele, treeName) for ele in numpy.loadtxt(fName, dtype = str, delimiter = "*"*100)]
            self.l_sample_weight[iSample] = weight
            self.l_sample_nJetMax[iSample] = nJetMax
        
        # Including the stop index (i.e. start --> stop)
        self.l_sample_treeIdx_start = [0] * len(samples)
        self.l_sample_treeIdx_stop = [-1] * len(samples)
        
        # Excluding the stop index (i.e. start --> stop-1)
        self.l_sample_treeEventBlockIdx_start = [[0] * len(ele) for ele in self.l_sample_fileAndTreeName]
        self.l_sample_treeEventBlockIdx_stop = [[-1] * len(ele) for ele in self.l_sample_fileAndTreeName]


class InputData :
    
    def __init__(
        self,
        nRow,
        nCol,
    ) :
        
        self.nRow = nRow
        self.nCol = nCol
        
        
        arr_idx = numpy.empty((nRow, nCol), dtype = numpy.uint32)
        arr_val = numpy.empty(nRow, dtype = numpy.float16)
        
        self.arr_idx_shared = multiprocessing.shared_memory.SharedMemory(create = True, size = arr_idx.nbytes)
        self.arr_val_shared = multiprocessing.shared_memory.SharedMemory(create = True, size = arr_val.nbytes)
        
        self.arr_idx = numpy.ndarray(arr_idx.shape, dtype = arr_idx.dtype, buffer = self.arr_idx_shared.buf)
        self.arr_val = numpy.ndarray(arr_val.shape, dtype = arr_val.dtype, buffer = self.arr_val_shared.buf)
        
        self.arr_idx_shape = arr_idx.shape
        self.arr_val_shape = arr_val.shape
        
        self.arr_idx_dtype = arr_idx.dtype
        self.arr_val_dtype = arr_val.dtype
    
    
    def close(self) :
        
        self.arr_idx_shared.close()
        self.arr_idx_shared.unlink()
        
        self.arr_val_shared.close()
        self.arr_val_shared.unlink()


def computeBins(
    fileAndTreeName,
    cut,
    xVar,
    yVar,
    layerInfo,
    entry_start,
    entry_stop,
    jetIdxOffset,
    nConsti,
    arr_idx_sharedBufName,
    arr_val_sharedBufName,
    arr_idx_shape,
    arr_val_shape,
    arr_idx_dtype,
    arr_val_dtype,
    fill_start,
    fill_end,
    debugId = None,
    printEvery = 1000,
) :
    
    if (entry_start >=  entry_stop) :
        
        return 0
    
    
    arr_idx_sharedBuf = multiprocessing.shared_memory.SharedMemory(name = arr_idx_sharedBufName)
    arr_val_sharedBuf = multiprocessing.shared_memory.SharedMemory(name = arr_val_sharedBufName)
    
    arr_idx_shared = numpy.ndarray(arr_idx_shape, dtype = arr_idx_dtype, buffer = arr_idx_sharedBuf.buf)
    arr_val_shared = numpy.ndarray(arr_val_shape, dtype = arr_val_dtype, buffer = arr_val_sharedBuf.buf)
    
    
    arr_idx = numpy.zeros((nConsti, 4), dtype = numpy.uint32)
    arr_val = numpy.zeros(nConsti, dtype = numpy.float16)
    
    
    jetIdx = -1
    nJet_sel = 0
    
    entryCount = 0
    
    xVarName = "xVar"
    yVarName = "yVar"
    wVarName = "wVar"
    constiCutVarName = "constiCutVar"
    resolverVarName = "resolverVar"
    
    d_branchName_alias = {
        xVarName: xVar,
        yVarName: yVar,
        wVarName: layerInfo.valueBranch,
    }
    
    if (layerInfo.cutBranch is not None) :
        
        d_branchName_alias[constiCutVarName] = layerInfo.cutBranch
    
    if (layerInfo.resolverBranch is not None) :
        
        d_branchName_alias[resolverVarName] = layerInfo.resolverBranch
    
    
    with uproot.open(fileAndTreeName) as tree :
        
        branches = tree.arrays(
            expressions = d_branchName_alias.keys(),
            cut = cut,
            aliases = d_branchName_alias,
            entry_start = entry_start,
            entry_stop = entry_stop,
        )
        
        nEvent = len(branches[xVarName])
        #nEvent = 1000
        
        
        for iEvent in range(0, nEvent) :
            
            nJet = len(branches[xVarName][iEvent])
            
            for iJet in range(0, nJet) :
                
                jetIdx += 1
                
                a_x  = branches[xVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                a_y  = branches[yVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                a_w  = branches[wVarName][iEvent][iJet].to_numpy()#.astype(dtype = numpy.float16)
                
                
                # Do NOT use these: the values are not cast properly
                ###a_x  = awkward.values_astype(branches[xVarName][iEvent][iJet], to = numpy.uint32)
                ###a_y  = awkward.values_astype(branches[yVarName][iEvent][iJet], to = numpy.uint32)
                ###a_w  = branches[wVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.float16)
                
                a_constiCut = None
                
                if (layerInfo.cutBranch is not None) :
                    
                    a_constiCut = branches[constiCutVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                    ###a_constiCut = awkward.values_astype(branches[constiCutVarName][iEvent][iJet], to = numpy.uint32)
                
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
                        numpy.uint32(jetIdxOffset+jetIdx),
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
                
                
                a_x  = None
                a_y  = None
                a_w  = None
                a_id = None
            
            #gc.collect()
            
            
            if (debugId is not None) :
                
                if (iEvent == 0 or iEvent+1 == nEvent or not (iEvent+1)%printEvery) :
                    
                    print("[%s] Processed event %d/%d." %(debugId, iEvent+1, nEvent))
    
    
    #print("%"*10, nJet_sel, len(d_arrayIdx))
    
    #return (arr_idx[0: entryCount], arr_val[0: entryCount])
    #return (fill_start, fill_end, arr_idx, arr_val)
    
    #return (nJet_sel, d_arrayIdx)
    
    
    #print("%"*10, arr_idx.shape)
    #print("%"*10, arr_val.shape)
    
    #print(jetIdxOffset, fill_start, fill_end, arr_idx)
    
    arr_idx_shared[fill_start: fill_end] = arr_idx
    arr_val_shared[fill_start: fill_end] = arr_val
    
    #print(arr_idx_shared)
    
    arr_idx_sharedBuf.close()
    arr_val_sharedBuf.close()
    
    return 0


#@memory_profiler.profile
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
        "--tag",
        help = "Tag (will create the training directory <tag>). If omitted, will use <datetime>.",
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--usedatetime",
        help = "If \"tag\" is provided, <datetime> will not be appended by default. Pass this flag to force adding <datetime>.",
        default = False,
        action = "store_true",
    )
    
    parser.add_argument(
        "--outdirbase",
        help = "Base output directory",
        type = str,
        required = False,
        default = "training_results",
    )
    
    
    args = parser.parse_args()
    d_args = vars(args)
    
    
    out_tag = "" if (args.tag is None) else args.tag.strip()
    
    if (not len(out_tag.strip())) :
        
        args.usedatetime = True
    
    datetime_tag = ""
    
    if (args.usedatetime) :
        
        datetime_tag = "%s" %(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    out_tag = "%s%s%s" %(out_tag, "_"*int(len(out_tag)*len(datetime_tag) > 0), datetime_tag)
    
    checkpoint_dir = "%s/model_checkpoints/%s" %(args.outdirbase, out_tag)
    tensorboard_dir = "%s/tensorboard/%s" %(args.outdirbase, out_tag)
    
    for dirname in [checkpoint_dir, tensorboard_dir] :
        
        if (os.path.exists(dirname)) :
            
            print("Error. Directory already exists: %s" %(dirname))
            exit(1)
    
    
    print("Model checkpoint dir: %s" %(checkpoint_dir))
    print("Tensorboard dir: %s" %(tensorboard_dir))
    
    
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
    
    brName_nConsti = d_loadConfig["constiBranch"]
    
    testFraction = d_loadConfig["testFraction"]
    nEventPerJob = d_loadConfig["loadEventPerJob"]
    
    l_layerInfo = []
    
    for iLayer, layer in enumerate(d_loadConfig["layers"]) :
        
        l_layerInfo.append(LayerInfo(
            layerIdx = iLayer,
            imgSpec = imgSpec,
            **layer,
        ))
    
    
    nLayer = len(l_layerInfo)
    img_shape = (imgSpec.nBinY, imgSpec.nBinX, nLayer)
    
    print("=====> Loading trees... Memory:", utils.getMemoryMB())
    
    d_catInfo_trn = sortedcontainers.SortedDict()
    d_catInfo_tst = sortedcontainers.SortedDict()
    
    for catNum in d_loadConfig["categories"] :
        
        category = d_loadConfig["categories"][catNum]
        
        d_catInfo_trn[catNum] = CategoryInfo(
            catNum = catNum,
            catName = category["name"],
            samples = category["samples"],
            l_layerInfo = l_layerInfo,
            cut = category["cut"],
        )
        
        d_catInfo_tst[catNum] = CategoryInfo(
            catNum = catNum,
            catName = category["name"],
            samples = category["samples"],
            l_layerInfo = l_layerInfo,
            cut = category["cut"],
        )
    
    
    nCategory = len(d_catInfo_trn)
    
    
    for (iCat, cat) in enumerate(d_catInfo_trn.keys()) :
        
        catInfo_trn = d_catInfo_trn[cat]
        catInfo_tst = d_catInfo_tst[cat]
        
        for iSample, sample_fileAndTreeName in enumerate(catInfo_trn.l_sample_fileAndTreeName):
            
            nJet_sample = 0
            
            for (iTree, fileAndTreeName) in enumerate(sample_fileAndTreeName) :
                
                print("Counting jets in: %s" %(fileAndTreeName))
                
                with uproot.open(fileAndTreeName) as tree :
                    
                    branches = tree.arrays(
                        expressions = [brName_nConsti],
                        cut = catInfo_trn.cut,
                        #entry_start = entry_start,
                        #entry_stop = entry_stop,
                    )
                    
                    a_nConsti = awkward.values_astype(branches[brName_nConsti], to = numpy.int64)
                    
                    nJet = awkward.count(a_nConsti)
                    nJet_sample += nJet
                
                if (nJet_sample >= catInfo_trn.l_sample_nJetMax[iSample]) :
                    
                    break
            
            if (nJet_sample < catInfo_trn.l_sample_nJetMax[iSample]) :
                
                catInfo_trn.l_sample_nJetMax[iSample] = nJet_sample
            
            catInfo_tst.l_sample_nJetMax[iSample] = int(testFraction * catInfo_trn.l_sample_nJetMax[iSample])
            catInfo_trn.l_sample_nJetMax[iSample] -= catInfo_tst.l_sample_nJetMax[iSample]
            
            
    
    
    print("=====> Reading branches... Memory:", utils.getMemoryMB())
    
    
    def read_data(d_catInfo) :
        
        nJetTot = 0
        nConstiTot = 0
        
        l_jetIdxOffset = []
        l_nConsti = []
        l_nConstiTot = [0]
        
        counter = -1
        
        for (iCat, cat) in enumerate(d_catInfo.keys()) :
            
            catInfo = d_catInfo[cat]
            
            for iSample, sample_fileAndTreeName in enumerate(catInfo.l_sample_fileAndTreeName):
                
                iTree = 0
                
                treeIdx_start = catInfo.l_sample_treeIdx_start[iSample]
                treeIdx_start = 0 if (treeIdx_start < 0) else treeIdx_start
                treeIdx_stop = catInfo.l_sample_treeIdx_stop[iSample] if (catInfo.l_sample_treeIdx_stop[iSample] >= 0) else len(sample_fileAndTreeName)-1
                
                for (iTree, fileAndTreeName) in enumerate(sample_fileAndTreeName) :
                    
                    if (iTree < treeIdx_start or iTree > treeIdx_stop) :
                        
                        continue
                    
                    nEvent = 0
                    
                    with uproot.open(fileAndTreeName) as tree :
                        
                        nEvent = tree.num_entries
                        
                        l_eventIdx = list(range(0, nEvent, nEventPerJob))
                        
                        if (l_eventIdx[-1] != nEvent) :
                            
                            l_eventIdx.append(nEvent)
                        
                        
                        eventBlockIdx_start = catInfo.l_sample_treeEventBlockIdx_start[iSample][iTree]
                        eventBlockIdx_start = 0 if (eventBlockIdx_start < 0) else eventBlockIdx_start
                        eventBlockIdx_stop = catInfo.l_sample_treeEventBlockIdx_stop[iSample][iTree] if (catInfo.l_sample_treeEventBlockIdx_stop[iSample][iTree] > 0) else len(l_eventIdx)-1
                        
                        idx = -1
                        
                        for idx in range(eventBlockIdx_start, eventBlockIdx_stop) :
                            
                            entry_start = l_eventIdx[idx]
                            entry_stop = l_eventIdx[idx+1]
                            
                            branches = tree.arrays(
                                expressions = [brName_nConsti],
                                cut = catInfo.cut,
                                entry_start = entry_start,
                                entry_stop = entry_stop,
                            )
                            
                            a_nConsti = awkward.values_astype(branches[brName_nConsti], to = numpy.int64)
                            
                            nJet = awkward.count(a_nConsti)
                            nConsti = awkward.sum(a_nConsti)
                            
                            for layerInfo in catInfo.l_layerInfo :
                                
                                counter += 1
                                #print("counter %d" %(counter))
                                
                                l_nConsti.append(nConsti)
                                
                                nConstiTot += nConsti
                                l_nConstiTot.append(nConstiTot)
                                
                                l_jetIdxOffset.append(nJetTot)
                                
                            
                            
                            nJetTot += nJet
                            catInfo.nJet += nJet
                            
                            catInfo.l_sample_nJet[iSample] += nJet
                            
                            if (catInfo.l_sample_nJet[iSample] >= catInfo.l_sample_nJetMax[iSample]) :
                                
                                break
                        
                        catInfo.l_sample_treeEventBlockIdx_stop[iSample][iTree] = idx+1
                    
                    
                    print("Preprocessed category %d (%d/%d), sample %d/%d, file %d/%d." %(
                        iCat, iCat+1, len(d_catInfo.keys()),
                        iSample+1, len(catInfo.l_sample_fileAndTreeName),
                        iTree+1, len(sample_fileAndTreeName)),
                    )
                    
                    
                    if (catInfo.l_sample_nJet[iSample] >= catInfo.l_sample_nJetMax[iSample]) :
                        
                        break
                
                catInfo.l_sample_treeIdx_stop[iSample] = iTree
        
        for (iCat, cat) in enumerate(d_catInfo.keys()) :
            
            catInfo = d_catInfo[cat]
            
            print(
                "Category %d: nJet %d %s (max %s)" %(
                iCat,
                catInfo.nJet,
                str(catInfo.l_sample_nJet),
                str(catInfo.l_sample_nJetMax),
            ))
        
        print("nJetTot:", nJetTot)
        print("nConstiTot:", nConstiTot)
        
        #exit()
        
        
        #arr_shuffledIdx = numpy.arange(0, nJetTot, dtype = numpy.int32)
        #numpy.random.shuffle(arr_shuffledIdx)
        
        #arr_label = numpy.zeros(nJetTot, dtype = int)
        
        
        l_job = []
        
        inpData = InputData(
            nRow = nConstiTot,
            nCol = 4,
        )
        
        counter = -1
        
        nCpu = multiprocessing.cpu_count()
        nCpu_frac = d_loadConfig["cpuFrac"]
        nCpu_use = int(nCpu_frac*nCpu)
        
        pool = multiprocessing.Pool(processes = nCpu_use, maxtasksperchild = 1)
        
        for (iCat, cat) in enumerate(d_catInfo.keys()) :
            
            catInfo = d_catInfo[cat]
            catNum = catInfo.catNum
            
            for iSample, sample_fileAndTreeName in enumerate(catInfo.l_sample_fileAndTreeName) :
                
                treeIdx_start = catInfo.l_sample_treeIdx_start[iSample]
                treeIdx_start = 0 if (treeIdx_start < 0) else treeIdx_start
                treeIdx_stop = catInfo.l_sample_treeIdx_stop[iSample] if (catInfo.l_sample_treeIdx_stop[iSample] >= 0) else len(sample_fileAndTreeName)-1
                #print("*"*10, iCat, iSample, treeIdx_start, treeIdx_stop)
                
                for (iTree, fileAndTreeName) in enumerate(sample_fileAndTreeName) :
                    
                    if (iTree < treeIdx_start or iTree > treeIdx_stop) :
                        
                        continue
                    
                    nEvent = 0
                    
                    with uproot.open(fileAndTreeName) as tree :
                        
                        nEvent = tree.num_entries
                        
                        l_eventIdx = list(range(0, nEvent, nEventPerJob))
                        
                        if (l_eventIdx[-1] != nEvent) :
                            
                            l_eventIdx.append(nEvent)
                        
                        eventBlockIdx_start = catInfo.l_sample_treeEventBlockIdx_start[iSample][iTree]
                        eventBlockIdx_start = 0 if (eventBlockIdx_start < 0) else eventBlockIdx_start
                        eventBlockIdx_stop = catInfo.l_sample_treeEventBlockIdx_stop[iSample][iTree] if (catInfo.l_sample_treeEventBlockIdx_stop[iSample][iTree] > 0) else len(l_eventIdx)-1
                        #print("#"*10, iCat, iSample, iTree, eventBlockIdx_start, eventBlockIdx_stop, len(l_eventIdx)-1)
                        
                        for idx in range(eventBlockIdx_start, eventBlockIdx_stop) :
                            
                            entry_start = l_eventIdx[idx]
                            entry_stop = l_eventIdx[idx+1]
                            
                            
                            for iLayer, layerInfo in enumerate(catInfo.l_layerInfo) :
                                
                                counter += 1
                                #print("counter %d, jetIdxOffset %d, fill_start %d, fill_end %d" %(counter, l_jetIdxOffset[counter], l_nConstiTot[counter], l_nConstiTot[counter+1]))
                                
                                fill_start = l_nConstiTot[counter]
                                fill_end = l_nConstiTot[counter+1]
                                
                                l_job.append(pool.apply_async(
                                    computeBins,
                                    (),
                                    dict(
                                        fileAndTreeName = fileAndTreeName,
                                        xVar = xVar,
                                        yVar = yVar,
                                        cut = catInfo.cut,
                                        layerInfo = layerInfo,
                                        entry_start = entry_start,
                                        entry_stop = entry_stop,
                                        jetIdxOffset = l_jetIdxOffset[counter],
                                        nConsti = l_nConsti[counter],
                                        arr_idx_sharedBufName = inpData.arr_idx_shared.name,
                                        arr_val_sharedBufName = inpData.arr_val_shared.name,
                                        arr_idx_shape = inpData.arr_idx_shape,
                                        arr_val_shape = inpData.arr_val_shape,
                                        arr_idx_dtype = inpData.arr_idx_dtype,
                                        arr_val_dtype = inpData.arr_val_dtype,
                                        fill_start = fill_start,
                                        fill_end = fill_end,
                                        debugId = "cat %d, sample %d/%d, tree %d/%d, layer %d/%d, job %d" %(
                                            catNum,
                                            iSample+1, len(catInfo.l_sample_fileAndTreeName),
                                            iTree+1, len(sample_fileAndTreeName),
                                            iLayer+1, len(catInfo.l_layerInfo),
                                            len(l_job)+1
                                        ),
                                        printEvery = nEventPerJob,
                                    ),
                                    #callback = inpData.fill,
                                ))
            
            
            print("\n%s Submitted all read-jobs for category %d %s\n" %("="*50, catNum, "="*50))
        
        pool.close()
        
        
        l_isJobDone = [False] * len(l_job)
        
        while(False in l_isJobDone) :
            
            for iJob, job in enumerate(l_job) :
                
                #print iJob, job
                
                if (job is None) :
                    
                    continue
                
                if (not l_isJobDone[iJob] and job.ready()) :
                    
                    l_isJobDone[iJob] = True
                    
                    retVal = job.get()
                    print("retVal", retVal)
                    
                    l_job[iJob] = None
                    
                    if (not sum(l_isJobDone) % 20) :
                        
                        gc.collect()
                    
                    print("Processed output of read-job num %d (%d/%d done) of category %d." %(iJob+1, sum(l_isJobDone), len(l_job), catNum))
                    print("=====> Memory:", utils.getMemoryMB())
        
        
        pool.join()
        
        print([d_catInfo[key].catNum for key in d_catInfo])
        print([d_catInfo[key].nJet for key in d_catInfo])
        print([d_catInfo[key].l_sample_nJet for key in d_catInfo])
        
        arr_label = numpy.repeat(
            [d_catInfo[key].catNum for key in d_catInfo],
            [d_catInfo[key].nJet for key in d_catInfo],
        )
        
        arr_weight = numpy.repeat(
            #[1.0/d_catInfo[key].nJet for key in d_catInfo],
            [1.0 for key in d_catInfo],
            
            [d_catInfo[key].nJet for key in d_catInfo],
        )
        
        nJetTot = len(arr_label)
        print("nJetTot = %d" %(nJetTot))
        
        #arr_idx = inpData.arr_idx
        #arr_val = inpData.arr_val
        #
        #nJetTot = len(arr_label)
        #
        #img_shape = (imgSpec.nBinY, imgSpec.nBinX, nLayer)
        #input_shape = (nJetTot,) + img_shape
        #
        ##arr_nonzero_idx = (arr_idx != [0, 0, 0, 0]).any(axis = 1).nonzero()
        #arr_nonzero_idx = arr_val.nonzero()[0]
        #arr_inputdata = tensorflow.sparse.reorder(tensorflow.sparse.SparseTensor(
        #    indices = arr_idx[arr_nonzero_idx],
        #    values = arr_val[arr_nonzero_idx],
        #    dense_shape = input_shape,
        #))
        #
        #arr_dense = tensorflow.sparse.to_dense(arr_inputdata).numpy()
        #print("arr_dense", arr_dense.shape)
        #
        #fig = matplotlib.pyplot.figure(figsize = [10, 8])
        #
        #axis = fig.add_subplot(1, 1, 1)
        #axis.set_aspect("equal", "box")
        #
        #img = axis.imshow(
        #    arr_dense.sum(axis = 0) / nJetTot,
        #    norm = matplotlib.colors.LogNorm(vmin = 1e-6, vmax = 1.0),
        #    origin = "lower",
        #    cmap = matplotlib.cm.get_cmap("nipy_spectral"),
        #)
        #
        #fig.colorbar(img, ax = axis)
        #
        #fig.tight_layout()
        #
        #fig.savefig("debug/test.pdf")
        
        return inpData, arr_label, arr_weight
    
    
    def get_dataset(
        inpData,
        arr_label,
        arr_weight,
    ) :
        
        arr_idx = inpData.arr_idx
        arr_val = inpData.arr_val
        
        
        print("=====> Read branches... Memory:", utils.getMemoryMB())
        
        nJetTot = len(arr_label)
        print("nJetTot = %d" %(nJetTot))
        
        img_shape = (imgSpec.nBinY, imgSpec.nBinX, nLayer)
        input_shape = (nJetTot,) + img_shape
        
        print("=====> Creating sparse tensor... Memory:", utils.getMemoryMB())
        
        #arr_nonzero_idx = (arr_idx != [0, 0, 0, 0]).any(axis = 1).nonzero()
        arr_nonzero_idx = arr_val.nonzero()[0]
        
        print(arr_idx)
        print(arr_idx[arr_nonzero_idx])
        
        arr_inputdata = tensorflow.sparse.reorder(tensorflow.sparse.SparseTensor(
            indices = arr_idx[arr_nonzero_idx],
            values = arr_val[arr_nonzero_idx],
            dense_shape = input_shape,
        ))
        
        print("=====> Created sparse tensor... Memory:", utils.getMemoryMB())
        print(arr_inputdata)
        
        print("=====> Freeing shared buffer... Memory:", utils.getMemoryMB())
        inpData.close()
        print("=====> Freed shared buffer... Memory:", utils.getMemoryMB())
        
        
        batch_size = d_loadConfig["batchSize"]
        nBatch = int(numpy.ceil(nJetTot/float(batch_size))) # + int(len(arr_shuffledIdx)%batch_size > 0)
        
        print("=====> Creating dataset... Memory:", utils.getMemoryMB())
        
        #arr_inputdata = tensorflow.sparse.to_dense(arr_inputdata)
        dataset_image = tensorflow.data.Dataset.from_tensor_slices(arr_inputdata)
        dataset_label = tensorflow.data.Dataset.from_tensor_slices(arr_label)
        dataset_weight = tensorflow.data.Dataset.from_tensor_slices(arr_weight)
        dataset = tensorflow.data.Dataset.zip((dataset_image, dataset_label, dataset_weight)).shuffle(buffer_size = nJetTot, reshuffle_each_iteration = False).batch(batch_size = batch_size, num_parallel_calls = tensorflow.data.AUTOTUNE).prefetch(tensorflow.data.AUTOTUNE)
        
        print("=====> Created dataset... Memory:", utils.getMemoryMB())
        print(dataset.element_spec)
        
        return (dataset, dataset_image, dataset_label, dataset_weight)
    
    
    # Training data
    inpData_trn, arr_label_trn, arr_weight_trn = read_data(d_catInfo = d_catInfo_trn)
    
    dataset_trn, dataset_image_trn, dataset_label_trn, dataset_weight_trn = get_dataset(
        inpData = inpData_trn,
        arr_label = arr_label_trn,
        arr_weight = arr_weight_trn,
    )
    
    
    # Testing data
    for (iCat, cat) in enumerate(d_catInfo_trn.keys()) :
        
        catInfo_trn = d_catInfo_trn[cat]
        catInfo_tst = d_catInfo_tst[cat]
        
        catInfo_tst.l_sample_treeIdx_start = catInfo_trn.l_sample_treeIdx_stop
        catInfo_tst.l_sample_treeEventBlockIdx_start = catInfo_trn.l_sample_treeEventBlockIdx_stop
    
    
    inpData_tst, arr_label_tst, arr_weight_tst = read_data(d_catInfo = d_catInfo_tst)
    
    dataset_tst, dataset_image_tst, dataset_label_tst, dataset_weight_tst = get_dataset(
        inpData = inpData_tst,
        arr_label = arr_label_tst,
        arr_weight = arr_weight_tst,
    )
    
    
    nJetTot_trn = len(arr_label_trn)
    nJetTot_tst = len(arr_label_tst)
    
    
    batch_size = d_loadConfig["batchSize"]
    nBatch_trn = int(numpy.ceil(nJetTot_trn/float(batch_size))) # + int(len(arr_shuffledIdx)%batch_size > 0)
    
    def print_nJet(d_catInfo) :
        
        for (iCat, cat) in enumerate(d_catInfo.keys()) :
            
            catInfo = d_catInfo[cat]
            
            print(
                "Category %d: nJet %d %s" %(
                cat,
                catInfo.nJet,
                str(catInfo.l_sample_nJet),
            ))
    
    print("nJetTot_trn:", nJetTot_trn, [d_catInfo_trn[key].nJet for key in d_catInfo_trn])
    print_nJet(d_catInfo_trn)
    print("nJetTot_tst:", nJetTot_tst, [d_catInfo_tst[key].nJet for key in d_catInfo_tst])
    print_nJet(d_catInfo_tst)
    
    
    ##### Dataset debug plot #####
    def plot_images(d_catInfo, dataset, suffix = "") :
        
        os.system("mkdir -p debug")
        
        print("="*10)
        
        l_imgArr = []
        
        l_count = []
        
        for (iCat, cat) in enumerate(d_catInfo.keys()) :
            
            l_imgArr.append([])
            l_count.append(0)
            
            for iLayer in range(0, nLayer) :
                
                l_imgArr[iCat].append(numpy.zeros((img_shape[0], img_shape[1])))
        
        
        for element in dataset.as_numpy_iterator():
            
            ##print(element.shape)
            #print(type(element[0]), element[0].shape)
            #print(type(element[1]), element[1])
            
            #catNum = element[1]
            #layerNum = element[0][-1]
            
            for (iCat, cat) in enumerate(d_catInfo.keys()) :
                
                l_catIdx = numpy.nonzero(element[1] == cat)[0]
                
                l_count[iCat] += len(l_catIdx)
                
                for iLayer in range(0, nLayer) :
                    
                    arr_sum_temp = element[0][l_catIdx, :, :, iLayer].sum(axis = 0)
                    
                    l_imgArr[iCat][iLayer] = l_imgArr[iCat][iLayer] + arr_sum_temp
                    
                    print(cat, arr_sum_temp.shape, l_imgArr[iCat][iLayer].shape, l_imgArr[iCat][iLayer].sum())
        
        
        
        for cat in d_catInfo.keys() :
            
            print("cat.nJet, l_count[iCat]:", d_catInfo[cat].nJet, l_count[cat])
            
            for iLayer in range(0, nLayer) :
                
                arr_img = l_imgArr[cat][iLayer]
                arr_img /= float(l_count[cat])
                
                fig = matplotlib.pyplot.figure(figsize = [10, 8])
                axis = fig.add_subplot(1, 1, 1)
                axis.set_aspect("equal", "box")
                img = axis.imshow(
                    arr_img,
                    norm = matplotlib.colors.LogNorm(vmin = 1e-6, vmax = 1.0),
                    origin = "lower",
                    cmap = matplotlib.cm.get_cmap("nipy_spectral"),
                )
                
                fig.suptitle("Category %d, Layer %d" %(cat, iLayer))
                
                fig.colorbar(img, ax = axis)
                
                fig.tight_layout()
                
                os.system("mkdir -p debug/%s" %(out_tag))
                fig.savefig("debug/%s/jetImage_cat%d_layer%d_fromTensorFlowDataset%s.pdf" %(out_tag, cat, iLayer, suffix))
                
                matplotlib.pyplot.close(fig)
    
    
    #plot_images(d_catInfo_trn, dataset_trn, suffix = "_trn")
    #plot_images(d_catInfo_tst, dataset_tst, suffix = "_tst")
    
    
    ##exit()
    
    network_name = d_loadConfig["network"]
    nEpoch = d_loadConfig["nEpoch"]
    
    model = networks.d_network[network_name](input_shape = img_shape, nCategory = nCategory)
    
    model_summary_str = []
    model.summary(print_fn = lambda x: model_summary_str.append(x))
    model_summary_str = "\n".join(model_summary_str)
    print(model_summary_str)
    
    
    #class MyLRSchedule(tensorflow.keras.optimizers.schedules.LearningRateSchedule) :
    #    
    #    def __init__(
    #        self,
    #        initial_learning_rate,
    #        loss_decrease_threshold,
    #        decay_rate,
    #    ) :
    #        
    #        self.initial_learning_rate = initial_learning_rate
    #        self.loss_decrease_threshold = loss_decrease_threshold
    #        self.decay_rate = decay_rate
    #    
    #    def __call__(self, step) :
    #        
    #        lr = self.initial_learning_rate / (step + 1)
    #        print("Learning rate (step %d) %0.4g" %(step+1, lr))
    #        
    #        return lr
    #    
    #    def get_config(self):
    #        
    #        return vars(self)
    #    
    #
    
    lr_boundaries = [ele * nBatch_trn for ele in d_loadConfig["learningRate"]["epoch"]]
    lr_values = d_loadConfig["learningRate"]["rate"]
    
    lr_schedule = tensorflow.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = lr_boundaries,
        values = lr_values,
    )
    
    print(lr_schedule)
    
    optimizer = tensorflow.keras.optimizers.Adam(
        learning_rate = lr_schedule,
    )
    
    
    tensorboard_file_writer_trn = tensorflow.summary.create_file_writer("%s/train" %(tensorboard_dir))
    tensorboard_file_writer_tst = tensorflow.summary.create_file_writer("%s/validation" %(tensorboard_dir))
    tensorboard_file_writer_params = tensorflow.summary.create_file_writer("%s/params" %(tensorboard_dir))
    tensorboard_file_writer_config = tensorflow.summary.create_file_writer("%s/config" %(tensorboard_dir))
    #tensorboard_file_writer.set_as_default()
    
    with tensorboard_file_writer_config.as_default() :
        
        tensorflow.summary.text(
            "configuration",
            "```\n%s\n```" %(d_loadConfig["fileContent"]),
            step = 0,
        )
        
        tensorflow.summary.text(
            "model",
            "```\n%s\n```" %(model_summary_str),
            step = 0,
        )
    
    
    def get_sample_table_str(d_catInfo) :
        
        l_table = []
        
        l_table.append([
            "Category",
            "Sample",
            "Jets",
        ])
        
        for (iCat, cat) in enumerate(d_catInfo.keys()) :
            
            catInfo = d_catInfo[cat]
            
            rowLabel = "%d (%s)" %(catInfo.catNum, catInfo.catName)
            
            for iSample, sample in enumerate(catInfo.l_sample) :
                
                l_table.append([
                    rowLabel if (not iSample) else "",
                    sample,
                    catInfo.l_sample_nJet[iSample],
                ])
            
            l_table.append([
                "**%s**" %(rowLabel),
                "**Total**",
                "**%d**" %(sum(catInfo.l_sample_nJet)),
            ])
        
        
        table_str = tabulate.tabulate(
            l_table,
            headers = "firstrow",
            #tablefmt = "fancy_grid"
            tablefmt = "github"
        )
        
        print(table_str)
        
        return table_str
    
    
    with tensorboard_file_writer_trn.as_default() :
        
        tensorflow.summary.text(
            "samples",
            get_sample_table_str(d_catInfo_trn),
            step = 0,
        )
    
    with tensorboard_file_writer_tst.as_default() :
        
        tensorflow.summary.text(
            "samples",
            get_sample_table_str(d_catInfo_tst),
            step = 0,
        )
    
    
    def my_loss(y_true, y_pred) :
        
        y_true = tensorflow.convert_to_tensor(y_true)
        y_pred = tensorflow.convert_to_tensor(y_pred)
        
        if (tensorflow.rank(y_true) == 1) :
            
            y_true = tensorflow.expand_dims(y_true, axis = 1)
            #print("y_true (after expand_dims):", y_true)
            #print("y_true.get_shape().as_list():", y_true.get_shape().as_list())
        
        dtype = y_pred.dtype
        nRow = y_true.get_shape().as_list()[0]
        
        #print("y_true:", y_true)
        
        y_true_onehot = tensorflow.one_hot(indices = tensorflow.gather(y_true, 0, axis = 1), depth = nCategory, dtype = dtype)
        #print("y_true_onehot:", y_true_onehot)
        
        y0_true_onehot = tensorflow.gather(y_true_onehot, 0, axis = 1)
        
        #print("y_pred:", y_pred)
        
        y0_pred = tensorflow.gather(y_pred, 0, axis = 1)
        #print("y0_pred:", y0_pred)
        
        D = 0.5*(tensorflow.math.tanh(50*(y0_pred-0.1)) + 1)
        D = tensorflow.expand_dims(D, axis = 1)
        #print("D:", D)
        
        ones = tensorflow.ones([nRow, 1], dtype = dtype)
        #print("ones:", ones)
        
        scale = tensorflow.concat([ones] + [D]*(nCategory-1), axis = 1)
        #print("scale:", scale)
        
        epsilon = tensorflow.keras.backend.epsilon()
        #epsilon = 1e-6
        
        loss_term1_tensor = (1.0/0.854) * (1-y0_true_onehot) * tensorflow.math.pow(y0_pred, 0.5) * tensorflow.math.log(1.0 - tensorflow.clip_by_value(y0_pred, epsilon, 1.0-epsilon))
        
        loss_term2_tensor = scale * y_true_onehot * tensorflow.math.log(tensorflow.clip_by_value(y_pred, epsilon, 1.0-epsilon))
        
        #print("loss_term1_tensor:", loss_term1_tensor)
        #print("loss_term2_tensor:", loss_term2_tensor)
        #print("tensorflow.math.reduce_sum(loss_term2_tensor, axis = 1):", tensorflow.math.reduce_sum(loss_term2_tensor, axis = 1))
        
        
        # Sum each row (event)
        loss_perrow = tensorflow.math.reduce_sum(loss_term2_tensor, axis = 1) + loss_term1_tensor
        #print("loss_perrow:", loss_perrow)
        
        # Average over all rows
        loss = -tensorflow.math.reduce_mean(loss_perrow)
        #print("loss:", loss)
        
        #print("loss:", loss)
        
        #if (numpy.isnan(loss.numpy()) or numpy.isinf(loss.numpy())) :
        #    
        #    print(tensorflow.where(tensorflow.math.is_nan(loss_tensor)))
        #    print(tensorflow.where(tensorflow.math.is_inf(loss_tensor)))
        #    print("tensorflow.math.reduce_min(y_pred):", tensorflow.math.reduce_min(y_pred))
        #    print("tensorflow.math.reduce_min(loss_tensor):", tensorflow.math.reduce_min(loss_tensor))
        #    
        #    print("y_pred nan:", tensorflow.gather(y_pred, indices = tensorflow.gather(tensorflow.where(tensorflow.math.is_nan(loss_tensor)), 0, axis = 1)))
        #    print("y_pred inf:", tensorflow.gather(y_pred, indices = tensorflow.gather(tensorflow.where(tensorflow.math.is_inf(loss_tensor)), 0, axis = 1)))
        #    
        #    #print(tensorflow.gather(scale, indices = tensorflow.gather(tensorflow.where(tensorflow.math.is_nan(scale)), 0, axis = 1)))
        #    
        #    print("tensorflow.math.reduce_sum(loss_tensor):", tensorflow.math.reduce_sum(loss_tensor))
        #    print("nRow:", nRow)
        #    
        #    #exit()
        
        
        return loss
    
    
    
    # Loss definition
    # no need to turn on from_logits if the network output is already a probability distribution like softmax
    loss_categorical_crossentropy = tensorflow.keras.losses.SparseCategoricalCrossentropy(
        from_logits = False
    )
    
    
    #loss_fn = loss_categorical_crossentropy
    loss_fn = my_loss
    
    
    
    # Accuracy definition
    sparse_categorical_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy()
    
    
    print("=====> Compiling model... Memory:", utils.getMemoryMB())
    
    model.compile(
        #optimizer = "adam",
        optimizer = optimizer,
        
        loss = loss_fn,
        #loss = my_loss,
        
        #metrics = ["accuracy"],
        metrics = [sparse_categorical_accuracy],
        
        run_eagerly = True,
    )
    
    print("=====> Compiled model... Memory:", utils.getMemoryMB())
    
    
    class CustomCallback(tensorflow.keras.callbacks.Callback):
        
        def __init__(self) :
            
            self.cb_dataset_trn = tensorflow.data.Dataset.zip(
                (dataset_image_trn, dataset_label_trn, dataset_weight_trn)
            ).batch(
                batch_size = batch_size,
                num_parallel_calls = tensorflow.data.AUTOTUNE
            ).prefetch(tensorflow.data.AUTOTUNE)
            
            self.cb_dataset_tst = tensorflow.data.Dataset.zip(
                (dataset_image_tst, dataset_label_tst, dataset_weight_tst)
            ).batch(
                batch_size = batch_size,
                num_parallel_calls = tensorflow.data.AUTOTUNE
            ).prefetch(tensorflow.data.AUTOTUNE)
            
            
            self.d_dataset_trn = {}
            self.d_dataset_tst = {}
            
            for cat in d_catInfo_trn.keys() :
                
                self.d_dataset_trn[cat] = tensorflow.data.Dataset.zip(
                    (dataset_image_trn, dataset_label_trn, dataset_weight_trn)
                ).filter(
                    lambda x, y, w: self.is_from_cat(x, y, cat)
                ).batch(
                    batch_size = batch_size,
                    num_parallel_calls = tensorflow.data.AUTOTUNE
                ).prefetch(tensorflow.data.AUTOTUNE)
                
                self.d_dataset_tst[cat] = tensorflow.data.Dataset.zip(
                    (dataset_image_tst, dataset_label_tst, dataset_weight_tst)
                ).filter(
                    lambda x, y, w: self.is_from_cat(x, y, cat)
                ).batch(
                    batch_size = batch_size,
                    num_parallel_calls = tensorflow.data.AUTOTUNE
                ).prefetch(tensorflow.data.AUTOTUNE)
            
            
            self.prev_epoch = 0
            self.prev_step = 0
            
            self.curr_epoch = 0
            self.curr_step = 0
            
            print("Initialized instance of CustomCallback.")
        
        
        def is_from_cat(self, x, y, catNum) :
            
            print(catNum, x, y, tensorflow.math.equal(y, catNum))
            #print(catNum, x)
            
            #return True
            return tensorflow.math.equal(y, catNum)
        
        
        def on_train_batch_end(self, batch, logs = None) :
            
            self.prev_step = nBatch_trn*self.prev_epoch + batch
            self.curr_step = nBatch_trn*self.curr_epoch + batch
            
            #print("%"*10, self.prev_step)
            #print(self.model.predict(self.d_dataset_trn[0]))
            #print("on_train_batch_end logs:", logs)
            
            if (logs is None) :
                
                return
            
            if (
                not self.prev_step or
                not (self.prev_step%d_loadConfig["batchLog"]) or
                self.prev_step == nBatch_trn-1
            ) :
                
                with tensorboard_file_writer_trn.as_default() :
                    
                    tensorflow.summary.scalar("batch_loss", logs["loss"], step = self.curr_step)
                    #tensorflow.summary.scalar("batch_accuracy", logs["accuracy"], step = self.curr_step)
                    tensorflow.summary.scalar("batch_accuracy", logs["sparse_categorical_accuracy"], step = self.curr_step)
                
                with tensorboard_file_writer_tst.as_default() :
                    
                    y_pred = self.model.predict(self.cb_dataset_tst)
                    
                    accuracy_fn = tensorflow.keras.metrics.SparseCategoricalAccuracy()
                    accuracy_fn.update_state(arr_label_tst, y_pred)
                    accuracy = accuracy_fn.result().numpy()
                    
                    loss = loss_fn(arr_label_tst, y_pred).numpy()
                    
                    tensorflow.summary.scalar("batch_loss", loss, step = self.curr_step)
                    tensorflow.summary.scalar("batch_accuracy", accuracy, step = self.curr_step)
        
        
        def on_epoch_begin(self, epoch, logs = None) :
            
            self.curr_epoch = epoch
        
        
        def on_epoch_end(self, epoch, logs = None) :
            
            self.prev_epoch = epoch
            
            with tensorboard_file_writer_trn.as_default() :
                
                y_pred = self.model.predict(self.cb_dataset_trn)
                
                accuracy_fn = tensorflow.keras.metrics.SparseCategoricalAccuracy()
                accuracy_fn.update_state(arr_label_trn, y_pred)
                accuracy = accuracy_fn.result().numpy()
                
                loss = loss_fn(arr_label_trn, y_pred).numpy()
                #print("my_epoch_loss (train):", loss)
                #
                #if (numpy.isnan(loss)) :
                #    
                #    print("Encountered nan in my_epoch_loss (train).")
                #    print(y_pred)
                
                tensorflow.summary.scalar("my_epoch_loss", loss, step = epoch)
                tensorflow.summary.scalar("my_epoch_accuracy", accuracy, step = epoch)
            
            
            with tensorboard_file_writer_tst.as_default() :
                
                y_pred = self.model.predict(self.cb_dataset_tst)
                
                accuracy_fn = tensorflow.keras.metrics.SparseCategoricalAccuracy()
                accuracy_fn.update_state(arr_label_tst, y_pred)
                accuracy = accuracy_fn.result().numpy()
                
                loss = loss_fn(arr_label_tst, y_pred).numpy()
                #print("my_epoch_loss (test):", loss)
                #
                #if (numpy.isnan(loss)) :
                #    
                #    print("Encountered nan in my_epoch_loss (test).")
                #    print(y_pred)
                
                tensorflow.summary.scalar("my_epoch_loss", loss, step = epoch)
                tensorflow.summary.scalar("my_epoch_accuracy", accuracy, step = epoch)
            
            
            
            with tensorboard_file_writer_params.as_default() :
                
                #tensorflow.summary.scalar("learning_rate", self.model.optimizer.learning_rate, step = epoch)
                tensorflow.summary.scalar("learning_rate", self.model.optimizer.learning_rate(self.prev_step), step = epoch)
            
            keys = list(logs.keys())
            #print(keys)
            #print("on_epoch_end logs:", logs)
            
            d_pred_trn = {}
            d_pred_tst = {}
            
            #print(
            #    "===== Epoch loss: train %0.4f, val %0.4f =====" %(
            #    loss_fn(
            #        numpy.array([ele for ele in dataset_label_trn.as_numpy_iterator()]),
            #        self.model.predict(
            #            tensorflow.data.Dataset.zip((dataset_image_trn, dataset_label_trn, dataset_weight_trn)).batch(batch_size = batch_size)
            #        )
            #    ).numpy(),
            #    
            #    loss_fn(
            #        numpy.array([ele for ele in dataset_label_tst.as_numpy_iterator()]),
            #        self.model.predict(
            #            tensorflow.data.Dataset.zip((dataset_image_tst, dataset_label_tst, dataset_weight_tst)).batch(batch_size = batch_size)
            #        )
            #    ).numpy(),
            #))
            
            #print(model.history.__dict__)
            
            #print("Trn pred shape", self.model.predict(dataset_trn).shape)
            #print("Tst pred shape", self.model.predict(dataset_tst).shape)
            
            for cat in d_catInfo_trn.keys() :
                
                pred_trn = self.model.predict(self.d_dataset_trn[cat])
                pred_tst = self.model.predict(self.d_dataset_tst[cat])
                
                d_pred_trn[cat] = pred_trn
                d_pred_tst[cat] = pred_tst
                
                print("Trn pred shape (cat%d)" %(cat), pred_trn.shape)
                print("Tst pred shape (cat%d)" %(cat), pred_tst.shape)
                
                #print(self.model.input)
                #print(self.model.output)
                #print("*"*10, cat)
                #print(pred_trn)
                
                for node in range(nCategory) :
                    
                    with tensorboard_file_writer_trn.as_default() :
                        
                        tensorflow.summary.histogram("output_node%d_cat%d" %(node, cat), pred_trn[:, node], step = epoch, buckets = 500)
                    
                    with tensorboard_file_writer_tst.as_default() :
                        
                        tensorflow.summary.histogram("output_node%d_cat%d" %(node, cat), pred_tst[:, node], step = epoch, buckets = 500)
            
            
            for iCat, cat_bkg in enumerate(range(1, nCategory)) :
                
                node_sig = 0
                node_bkg = cat_bkg
                cat_sig = 0
                
                pred_sig_trn = d_pred_trn[cat_sig][:, node_sig] / (d_pred_trn[cat_sig][:, node_sig] + d_pred_trn[cat_sig][:, node_bkg])
                pred_bkg_trn = d_pred_trn[cat_bkg][:, node_sig] / (d_pred_trn[cat_bkg][:, node_sig] + d_pred_trn[cat_bkg][:, node_bkg])
                
                pred_sig_tst = d_pred_tst[cat_sig][:, node_sig] / (d_pred_tst[cat_sig][:, node_sig] + d_pred_tst[cat_sig][:, node_bkg])
                pred_bkg_tst = d_pred_tst[cat_bkg][:, node_sig] / (d_pred_tst[cat_bkg][:, node_sig] + d_pred_tst[cat_bkg][:, node_bkg])
                
                with tensorboard_file_writer_trn.as_default() :
                    
                    tensorflow.summary.histogram("classifier_cat%dvs%d_cat%d" %(cat_sig, cat_bkg, cat_sig), pred_sig_trn, step = epoch, buckets = 500)
                    tensorflow.summary.histogram("classifier_cat%dvs%d_cat%d" %(cat_sig, cat_bkg, cat_bkg), pred_bkg_trn, step = epoch, buckets = 500)
                
                with tensorboard_file_writer_tst.as_default() :
                    
                    tensorflow.summary.histogram("classifier_cat%dvs%d_cat%d" %(cat_sig, cat_bkg, cat_sig), pred_sig_tst, step = epoch, buckets = 500)
                    tensorflow.summary.histogram("classifier_cat%dvs%d_cat%d" %(cat_sig, cat_bkg, cat_bkg), pred_bkg_tst, step = epoch, buckets = 500)
                
                
                # Train
                arr_eff_bkg_trn, arr_eff_sig_trn, arr_threshold_trn = sklearn.metrics.roc_curve(
                    y_true = numpy.repeat(
                        [cat_sig, cat_bkg],
                        [len(pred_sig_trn), len(pred_bkg_trn)],
                    ),
                    y_score = numpy.concatenate(
                        [
                            pred_sig_trn,
                            pred_bkg_trn,
                        ],
                        axis = None,
                    ),
                    pos_label = cat_sig,
                )
                
                auc_trn = sklearn.metrics.auc(arr_eff_bkg_trn, arr_eff_sig_trn)
                
                
                # Test
                arr_eff_bkg_tst, arr_eff_sig_tst, arr_threshold_tst = sklearn.metrics.roc_curve(
                    y_true = numpy.repeat(
                        [cat_sig, cat_bkg],
                        [len(pred_sig_tst), len(pred_bkg_tst)],
                    ),
                    y_score = numpy.concatenate(
                        [
                            pred_sig_tst,
                            pred_bkg_tst,
                        ],
                        axis = None,
                    ),
                    pos_label = cat_sig,
                )
                
                auc_tst = sklearn.metrics.auc(arr_eff_bkg_tst, arr_eff_sig_tst)
                
                
                with tensorboard_file_writer_trn.as_default() :
                    
                    tensorflow.summary.scalar("auc_cat%d_vs_cat%d" %(cat_sig, cat_bkg), auc_trn, step = epoch)
                    
                    tfimage = utils_tensorflow.get_tfimage_xyplot(
                        l_plotdata = [
                            [
                                arr_eff_sig_trn, arr_eff_bkg_trn, "-", {}
                            ],
                        ],
                        xlim = (0, 1),
                        ylim = (1e-6, 1),
                        logy = True,
                        xlabel = "cat%d efficiency" %(cat_sig),
                        ylabel = "cat%d efficiency" %(cat_bkg),
                    )
                    tensorflow.summary.image("roc_cat%d_vs_cat%d" %(cat_sig, cat_bkg), tfimage, step = epoch)
                    
                    
                    tfimage = utils_tensorflow.get_tfimage_xyplot(
                        l_plotdata = [
                            [
                                arr_threshold_trn, arr_eff_sig_trn, "r-",
                                {
                                    "label": "cat%d" %(cat_sig),
                                },
                            ],
                            [
                                arr_threshold_trn, arr_eff_bkg_trn, "b-",
                                {
                                    "label": "cat%d" %(cat_bkg),
                                },
                            ],
                        ],
                        xlim = (0, 1),
                        ylim = (1e-6, 1),
                        logy = True,
                        xlabel = "Cut on cat%d_vs_cat%d classifier" %(cat_sig, cat_bkg),
                        ylabel = "Efficiency",
                    )
                    tensorflow.summary.image("eff_cat%d_vs_cat%d" %(cat_sig, cat_bkg), tfimage, step = epoch)
                
                with tensorboard_file_writer_tst.as_default() :
                    
                    tensorflow.summary.scalar("auc_cat%d_vs_cat%d" %(cat_sig, cat_bkg), auc_tst, step = epoch)
                    
                    tfimage = utils_tensorflow.get_tfimage_xyplot(
                        l_plotdata = [
                            [
                                arr_eff_sig_tst, arr_eff_bkg_tst, "-", {}
                            ],
                        ],
                        xlim = (0, 1),
                        ylim = (1e-6, 1),
                        logy = True,
                        xlabel = "cat%d efficiency" %(cat_sig),
                        ylabel = "cat%d efficiency" %(cat_bkg),
                    )
                    tensorflow.summary.image("roc_cat%d_vs_cat%d" %(cat_sig, cat_bkg), tfimage, step = epoch)
                    
                    
                    tfimage = utils_tensorflow.get_tfimage_xyplot(
                        l_plotdata = [
                            [
                                arr_threshold_tst, arr_eff_sig_tst, "r-",
                                {
                                    "label": "cat%d" %(cat_sig),
                                },
                            ],
                            [
                                arr_threshold_tst, arr_eff_bkg_tst, "b-",
                                {
                                    "label": "cat%d" %(cat_bkg),
                                },
                            ],
                        ],
                        xlim = (0, 1),
                        ylim = (1e-6, 1),
                        logy = True,
                        xlabel = "Cut on cat%d_vs_cat%d classifier" %(cat_sig, cat_bkg),
                        ylabel = "Efficiency",
                    )
                    tensorflow.summary.image("eff_cat%d_vs_cat%d" %(cat_sig, cat_bkg), tfimage, step = epoch)
    
    
    
    checkpoint_file = "%s/weights_epoch-{epoch:d}.hdf5" %(checkpoint_dir)
    #model_save_dir = "%s/model_epoch-{epoch:d}" %(checkpoint_dir)
    
    checkpoint_callback  =  tensorflow.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_file,
        ##filepath = model_save_dir,
        monitor = "val_loss",
        verbose = 0,
        save_best_only = False,
        save_weights_only = False,
        mode = "auto",
        save_freq = "epoch",
        #options = None,
        #**kwargs
    )
    
    
    my_callback = CustomCallback()
    
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
        log_dir = tensorboard_dir,
        histogram_freq = 1,
        write_graph = True,
        write_images = False,
        write_steps_per_second = False,
        update_freq = "epoch",
        #update_freq = 10,
        #profile_batch = (1, nBatch_trn),
        #profile_batch = nBatch_trn,
        #profile_batch = 10,
        embeddings_freq = 0,
        embeddings_metadata = None,
    )
    
    
    
    
    print("=====> Starting fit... Memory:", utils.getMemoryMB())
    
    history = model.fit(
        x = dataset_trn,
        epochs = nEpoch,
        #batch_size = batch_size,
        validation_data = dataset_tst,
        shuffle = False,
        #max_queue_size = 20,
        #use_multiprocessing = True,
        #workers = nCpu_use,
        callbacks = [checkpoint_callback, tensorboard_callback, my_callback]
    )



if (__name__ == "__main__") :
    
    main()


#from torch.utils.tensorboard import SummaryWriter 
#writer = SummaryWriter() 
#writer.add_scalar(“Output”, output, iter_num) 
