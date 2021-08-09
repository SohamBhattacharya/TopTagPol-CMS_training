# https://sungwookyoo.github.io/tips/ArgParser/


from __future__ import print_function

#import mxnet
import awkward
import collections
#import cppyy
#import cppyy.ll
import concurrent.futures
import datetime
import gc
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
import psutil
import pympler
import sortedcontainers
import sparse
import sys
import tensorflow
import time
import uproot

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

import utils



# https://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_10_5_0/doc/html/dc/d55/classreco_1_1PFCandidate.html#ac55e8d12f21ad7c3fb6400f7fa60690c
d_pdgid2layer = {
    +211     : numpy.uint32(0),
    -211     : numpy.uint32(0),
    
    +11      : numpy.uint32(1),
    -11      : numpy.uint32(1),
    
    +13      : numpy.uint32(2),
    -13      : numpy.uint32(2),
    
    +22      : numpy.uint32(3),
    -22      : numpy.uint32(3),
    
    +130     : numpy.uint32(4),
    -130     : numpy.uint32(4),
    
    +1       : numpy.uint32(5),
    -1       : numpy.uint32(5),
    
    +2       : numpy.uint32(6),
    -2       : numpy.uint32(6),
    
    0        : numpy.uint32(7),
}

vec_pdgid2layer = numpy.vectorize(d_pdgid2layer.get)


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
        valueBranch,
        cutBranch,
        resolverOperation,
        resolverUpdate = None,
        resolverBranch = None,
    ) :
        
        self.layerIdx = layerIdx
        
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
        l_fileAndTreeName,
        l_layerInfo,
        cut,
    ) :
        
        self.catNum = catNum
        self.l_fileAndTreeName = l_fileAndTreeName
        self.l_layerInfo = l_layerInfo
        self.cut = cut
        
        self.nJet = 0


class TrainingData :
    
    def __init__(
        self,
        nRow,
        nCol,
    ) :
        
        self.nRow = nRow
        self.nCol = nCol
        
        
        arr_idx = numpy.empty((nRow, 4), dtype = numpy.uint32)
        arr_val = numpy.empty(nRow, dtype = numpy.float16)
        
        self.arr_idx_shared = multiprocessing.shared_memory.SharedMemory(create = True, size = arr_idx.nbytes)
        self.arr_val_shared = multiprocessing.shared_memory.SharedMemory(create = True, size = arr_val.nbytes)
        
        self.arr_idx = numpy.ndarray(arr_idx.shape, dtype = arr_idx.dtype, buffer = self.arr_idx_shared.buf)
        self.arr_val = numpy.ndarray(arr_val.shape, dtype = arr_val.dtype, buffer = self.arr_val_shared.buf)
        
        self.arr_idx_shape = arr_idx.shape
        self.arr_val_shape = arr_val.shape
        
        self.arr_idx_dtype = arr_idx.dtype
        self.arr_val_dtype = arr_val.dtype
    
    
    #def fill(self, result) :
    #    
    #    fill_start, fill_end, arr_idx_temp, arr_val_temp = result
    #    
    #    self.arr_idx[fill_start: fill_end] = arr_idx_temp
    #    self.arr_val[fill_start: fill_end] = arr_val_temp
    #    
    #    result = None
    #    fill_start = None
    #    fill_end = None
    #    arr_idx_temp = None
    #    arr_val_temp = None
    #    
    #    del result
    #    del fill_start
    #    del fill_end
    #    del arr_idx_temp
    #    del arr_val_temp
    #    
    #    #cppyy.ll.free(arr_idx_temp)
    #    #cppyy.ll.free(arr_val_temp)
    #    
    #    print("size(arr_idx) in MB:", self.arr_idx.nbytes/1024.0**2)
    #    print("size(arr_val) in MB:", self.arr_val.nbytes/1024.0**2)
    #    
    #    gc.collect()
    
    
    def close(self) :
        
        self.arr_idx_shared.close()
        self.arr_idx_shared.unlink()
        
        self.arr_val_shared.close()
        self.arr_val_shared.unlink()


def getMemoryMB(process = -1) :
    
    if (process < 0) :
        
        process = psutil.Process(os.getpid())
    
    mem = process.memory_info().rss / 1024.0**2
    
    return mem


def pdgid2layer(pdgid) :
    
    pdgid = abs(pdgid)
    
    if (pdgid == 211) :
        return 0
        
    if (pdgid == 11) :
        return 1
        
    if (pdgid == 13) :
        return 2
        
    if (pdgid == 22) :
        return 3
        
    if (pdgid == 130) :
        return 4
        
    if (pdgid == 1) :
        return 5
        
    if (pdgid == 2) :
        return 6
        
    if (pdgid == 0) :
        return 7
    
    return 7





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
                
                #a_x  = branches[xVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                #a_y  = branches[yVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                #a_w  = branches[wVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.float16)
                
                a_x  = awkward.values_astype(branches[xVarName][iEvent][iJet], to = numpy.uint32)
                a_y  = awkward.values_astype(branches[yVarName][iEvent][iJet], to = numpy.uint32)
                a_w  = branches[wVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.float16)
                
                a_constiCut = None
                
                if (layerInfo.cutBranch is not None) :
                    
                    a_constiCut = branches[constiCutVarName][iEvent][iJet].to_numpy().astype(dtype = numpy.uint32)
                    #a_constiCut = awkward.values_astype(branches[constiCutVarName][iEvent][iJet], to = numpy.uint32)
                
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
    
    imgSpec = ImageSpec(
        nBinX = 50,
        nBinY = 50,
        
        xMin = -1.0,
        xMax = +1.0,
        
        yMin = -1.0,
        yMax = +1.0,
    )
    
    xVar = "(jet_selectedPatJetsAK15PFPuppi_constiTrans_x_reco - {xMin}) / {xBinWidth}".format(
        xMin = imgSpec.xMin,
        xBinWidth = imgSpec.xBinWidth,
    )
    
    yVar = "(jet_selectedPatJetsAK15PFPuppi_constiTrans_y_reco - {yMin}) / {yBinWidth}".format(
        yMin = imgSpec.yMin,
        yBinWidth = imgSpec.yBinWidth,
    )
    
    brName_nConsti = "jet_selectedPatJetsAK15PFPuppi_nConsti_reco"
    
    l_layerInfo = []
    
    for pdgid in [11, 13, 22, 211, 130] :
        
        cutBranch = "abs(jet_selectedPatJetsAK15PFPuppi_consti_id_reco) == {pdgid}".format(
            pdgid = pdgid
        )
        
        l_layerInfo.append(LayerInfo(
            layerIdx = len(l_layerInfo),
            valueBranch = "jet_selectedPatJetsAK15PFPuppi_constiTrans_w_reco",
            cutBranch = cutBranch,
            resolverOperation = "{newValue}+{oldValue}",
        ))
    
    for pdgid in [11, 13, 211] :
        
        cutBranch = "(jet_selectedPatJetsAK15PFPuppi_consti_dxy_reco > 0) * (jet_selectedPatJetsAK15PFPuppi_consti_pT_reco > 20) * (abs(jet_selectedPatJetsAK15PFPuppi_consti_id_reco) == {pdgid})".format(
            pdgid = pdgid
        )
        
        l_layerInfo.append(LayerInfo(
            layerIdx = len(l_layerInfo),
            valueBranch = "1.0 - (min(3, jet_selectedPatJetsAK15PFPuppi_consti_dxy_reco)/3.0)",
            cutBranch = cutBranch,
            resolverOperation = "{newValue} if ({newResolver} > {oldResolver}) else {oldValue}",
            resolverUpdate = "max({newResolver}, {oldResolver})",
            resolverBranch = "jet_selectedPatJetsAK15PFPuppi_consti_pT_reco",
        ))
    
    
    nLayer = len(l_layerInfo)
    
    print("=====> Loading trees... Memory:", getMemoryMB())
    
    d_catInfo = sortedcontainers.SortedDict()
    
    d_catInfo[0] = CategoryInfo(
        catNum = 0,
        l_fileAndTreeName = [
            "/nfs/dust/cms/user/sobhatta/work/TopTagPol/TreeMaker/CMSSW_10_5_0/src/ntupleTree_ZprimeToTT_M1000_W10_new.root:treeMaker/tree",
        ]*300,
        l_layerInfo = l_layerInfo,
        cut = "(jet_selectedPatJetsAK15PFPuppi_pT_reco > 200) & (abs(jet_selectedPatJetsAK15PFPuppi_eta_reco) < 2.4) & (jet_selectedPatJetsAK15PFPuppi_nConsti_reco >= 3) & (jet_selectedPatJetsAK15PFPuppi_nearestGenTopDR_reco < 1) & (jet_selectedPatJetsAK15PFPuppi_nearestGenTopIsLeptonic_reco > 0.5)",
    )
    
    d_catInfo[1] = CategoryInfo(
        catNum = 1,
        l_fileAndTreeName = [
            "/nfs/dust/cms/user/sobhatta/work/TopTagPol/TreeMaker/CMSSW_10_5_0/src/ntupleTree_QCD_Pt_470to600_PREMIX_RECODEBUG_new.root:treeMaker/tree",
    ]*100,
        l_layerInfo = l_layerInfo,
        cut = "(jet_selectedPatJetsAK15PFPuppi_pT_reco > 200) & (abs(jet_selectedPatJetsAK15PFPuppi_eta_reco) < 2.4) & (jet_selectedPatJetsAK15PFPuppi_nConsti_reco >= 3) & (jet_selectedPatJetsAK15PFPuppi_nearestGenTopDR_reco > 1)",
    )
    
    
    nCategory = len(d_catInfo)
    
    
    print("=====> Loaded trees... Memory:", getMemoryMB())
    
    
    jetIdx = 0
    
    l_idx = []
    l_val = []
    l_label = []
    
    d_sortedIdx = sortedcontainers.SortedDict()
    
    print("=====> Reading branches... Memory:", getMemoryMB())
    
    nEventPerJob = 10000
    
    
    sharedList_arrayIdx = multiprocessing.Manager().list()
    
    
    l_sparseArr_idx = []
    l_sparseArr_val = []
    
    nJetTot = 0
    nConstiTot = 0
    
    l_jetIdxOffset = []
    l_nConsti = []
    l_nConstiTot = [0]
    
    for (iCat, cat) in enumerate(d_catInfo.keys()) :
        
        catInfo = d_catInfo[cat]
        
        for (iTree, fileAndTreeName) in enumerate(catInfo.l_fileAndTreeName) :
            
            nEvent = 0
            
            with uproot.open(fileAndTreeName) as tree :
                
                nEvent = tree.num_entries
                
                l_eventIdx = list(range(0, nEvent, nEventPerJob))
                
                if (l_eventIdx[-1] != nEvent) :
                    
                    l_eventIdx.append(nEvent)
                
                for idx in range(0, len(l_eventIdx)-1) :
                    
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
                        
                        #branch_alias = "var"
                        #
                        #branches = tree.arrays(
                        #    expressions = [branch_alias],
                        #    aliases = {branch_alias: layerInfo.cutBranch},
                        #    cut = catInfo.cut,
                        #    entry_start = entry_start,
                        #    entry_stop = entry_stop,
                        #    language = utils.uproot_lang,
                        #)
                        #
                        #nConsti = awkward.count_nonzero(branches[branch_alias], axis = None)
                        l_nConsti.append(nConsti)
                        
                        nConstiTot += nConsti
                        l_nConstiTot.append(nConstiTot)
                        
                        l_jetIdxOffset.append(nJetTot)
                        
                    
                    
                    nJetTot += nJet
                    catInfo.nJet += nJet
            
            
            print("Preprocessed category %d/%d file %d/%d." %(iCat+1, len(d_catInfo.keys()), iTree+1, len(catInfo.l_fileAndTreeName)))
    
    print("nJetTot:", nJetTot)
    print("nConstiTot:", nConstiTot)
    
    l_job = []
    #l_catNum = []
    #l_nJet = []
    #l_nConsti = []
    #l_constiIdx = []
    
    trnData = TrainingData(
        nRow = nConstiTot,
        nCol = 4,
    )
    
    counter = -1
    #nJetTot = 0
    #nConstiTot = 0
    
    nCpu = multiprocessing.cpu_count()
    nCpu_frac = 0.99
    nCpu_use = int(nCpu_frac*nCpu)
    
    pool = multiprocessing.Pool(processes = nCpu_use, maxtasksperchild = 1)
    
    for (iCat, cat) in enumerate(d_catInfo.keys()) :
        
        catInfo = d_catInfo[cat]
        catNum = catInfo.catNum
        
        for (iTree, fileAndTreeName) in enumerate(catInfo.l_fileAndTreeName) :
            
            nEvent = 0
            
            with uproot.open(fileAndTreeName) as tree :
                
                nEvent = tree.num_entries
                
                l_eventIdx = list(range(0, nEvent, nEventPerJob))
                
                if (l_eventIdx[-1] != nEvent) :
                    
                    l_eventIdx.append(nEvent)
                
                for idx in range(0, len(l_eventIdx)-1) :
                    
                    entry_start = l_eventIdx[idx]
                    entry_stop = l_eventIdx[idx+1]
                    
                    #branches = tree.arrays(
                    #    expressions = [brName_nConsti],
                    #    cut = catInfo.cut,
                    #    entry_start = entry_start,
                    #    entry_stop = entry_stop,
                    #)
                    #
                    #nJet = awkward.count(branches[brName_nConsti])
                    
                    for iLayer, layerInfo in enumerate(catInfo.l_layerInfo) :
                        
                        #branch_alias = "var"
                        #
                        #branches = tree.arrays(
                        #    expressions = [branch_alias],
                        #    aliases = {branch_alias: layerInfo.cutBranch},
                        #    cut = catInfo.cut,
                        #    entry_start = entry_start,
                        #    entry_stop = entry_stop,
                        #    language = utils.uproot_lang,
                        #)
                        #
                        #nConsti = awkward.count_nonzero(branches[branch_alias], axis = None)
                        
                        counter += 1
                        
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
                                arr_idx_sharedBufName = trnData.arr_idx_shared.name,
                                arr_val_sharedBufName = trnData.arr_val_shared.name,
                                arr_idx_shape = trnData.arr_idx_shape,
                                arr_val_shape = trnData.arr_val_shape,
                                arr_idx_dtype = trnData.arr_idx_dtype,
                                arr_val_dtype = trnData.arr_val_dtype,
                                fill_start = fill_start,
                                fill_end = fill_end,
                                debugId = "cat %d, tree %d/%d, layer %d/%d, job %d" %(catNum, iTree+1, len(catInfo.l_fileAndTreeName), iLayer+1, len(catInfo.l_layerInfo), len(l_job)+1),
                                printEvery = nEventPerJob,
                            ),
                            #callback = trnData.fill,
                        ))
                        
                        
                        #l_nConsti.append(nConsti)
                        #l_constiIdx.append(nConstiTot)
                        
                        #nConstiTot += nConsti
                    
                    
                    #l_catNum.append(catNum)
                    #l_nJet.append(nJet)
                    
                    #catInfo.nJet += nJet
                    #nJetTot += nJet
        
        
        print("\n%s Submitted all read-jobs for category %d %s\n" %("="*50, catNum, "="*50))
        
        #pool.close()
        #pool.join()
    
    
    
    #print("nJetTot:", nJetTot)
    #print("nConstiTot:", nConstiTot)
    
    pool.close()
    #pool.join()
    
    #arr_idx = numpy.zeros((nConstiTot, 4), dtype = numpy.int64)
    #arr_val = numpy.zeros(nConstiTot, dtype = numpy.float16)
    #
    ##time.sleep(10)
    ##exit()
    
    
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
                print("=====> Memory:", getMemoryMB())
    
    
    pool.join()
    
    
    arr_idx = trnData.arr_idx
    arr_val = trnData.arr_val
    
    
    
    # Fill the labels
    l_label = numpy.repeat(
        [d_catInfo[key].catNum for key in d_catInfo],
        [d_catInfo[key].nJet for key in d_catInfo],
    )
    
    print([d_catInfo[key].catNum for key in d_catInfo])
    print([d_catInfo[key].nJet for key in d_catInfo])
    
    
    print("=====> Read branches... Memory:", getMemoryMB())
    
    
    #l_label = numpy.array(l_label)
    
    nJetTot = len(l_label)
    print("nJetTot = %d (%d)" %(nJetTot, jetIdx))
    
    input_shape = (nJetTot, imgSpec.nBinY, imgSpec.nBinX, nLayer)
    img_shape = (imgSpec.nBinY, imgSpec.nBinX, nLayer)
    
    print("=====> Creating sparse tensor... Memory:", getMemoryMB())
    
    #arr_nonzero_idx = (arr_idx != [0, 0, 0, 0]).any(axis = 1).nonzero()
    arr_nonzero_idx = arr_val.nonzero()[0]
    
    print(arr_idx)
    print(arr_idx[arr_nonzero_idx])
    
    print(arr_idx[arr_nonzero_idx[1205]], arr_idx[arr_nonzero_idx[1206]], arr_idx[arr_nonzero_idx[1207]])
    
    input_array = tensorflow.sparse.reorder(tensorflow.sparse.SparseTensor(
        indices = arr_idx[arr_nonzero_idx],
        values = arr_val[arr_nonzero_idx],
        dense_shape = input_shape,
    ))
    
    #input_array = sparse.COO(
    #    coords = arr_idx[arr_nonzero_idx].T,
    #    data = arr_val[arr_nonzero_idx],
    #    shape = input_shape
    #)
    
    
    print("=====> Created sparse tensor... Memory:", getMemoryMB())
    
    print(input_array)
    
    #print("=====> Freeing dictionary... Memory:", getMemoryMB())
    ##d_sortedIdx = None
    #del d_sortedIdx
    #del l_sparseArr_idx
    #del l_sparseArr_val
    #del arr_idx
    #del arr_val
    #del arr_nonzero_idx
    #del trnData
    #gc.collect()
    #print("=====> Freed dictionary... Memory:", getMemoryMB())
    
    
    print("=====> Freeing shared buffer... Memory:", getMemoryMB())
    trnData.close()
    print("=====> Freed shared buffer... Memory:", getMemoryMB())
    
    #print("Waiting...")
    #time.sleep(50)
    
    
    ##### Debug plot #####
    
    ##startIdx = 0
    ###a_temp = tensorflow.make_ndarray(tensorflow.sparse.to_dense(input_array))
    ###arr_dense = tensorflow.sparse.to_dense(input_array).numpy()
    ##
    ##arr_dense = input_array.todense()
    ##
    ##print(arr_dense.shape)
    ##
    ##for cat in d_catInfo :
    ##    
    ##    endIdx = startIdx + d_catInfo[cat].nJet
    ##    
    ##    a_temp = arr_dense[startIdx: endIdx] / d_catInfo[cat].nJet
    ##    a_temp = a_temp.sum(axis = 0).sum(axis = -1)
    ##    print(a_temp.shape)
    ##    
    ##    startIdx = endIdx
    ##    
    ##    
    ##    fig = matplotlib.pyplot.figure(figsize = [10, 8])
    ##    
    ##    axis = fig.add_subplot(1, 1, 1)
    ##    axis.set_aspect("equal", "box")
    ##    
    ##    img = axis.imshow(
    ##        a_temp,
    ##        norm = matplotlib.colors.LogNorm(vmin = 1e-6, vmax = 1.0),
    ##        origin = "lower",
    ##        cmap = matplotlib.cm.get_cmap("nipy_spectral"),
    ##    )
    ##    
    ##    fig.colorbar(img, ax = axis)
    ##    
    ##    fig.tight_layout()
    ##    
    ##    fig.savefig("jetImage_cat%d.pdf" %(d_catInfo[cat].catNum))
    
    
    a_shuffledIdx = numpy.arange(0, len(l_label), dtype = numpy.int32)
    numpy.random.shuffle(a_shuffledIdx)
    #a_shuffledIdx = numpy.reshape(a_shuffledIdx, (len(a_shuffledIdx), 1))
    
    batch_size = 1000
    nBatch = int(numpy.ceil(len(a_shuffledIdx)/float(batch_size))) # + int(len(a_shuffledIdx)%batch_size > 0)
    
    a_shuffledIdx_batch = numpy.array_split(a_shuffledIdx, nBatch)
    
    
    
    def generator() :
        
        #iJet = -1
        #
        #for jet in input_array :
        #    
        #    iJet += 1
        #    
        #    dense = jet.todense()
        #    #print("dense", dense)
        #    #print("dense shape", dense.shape)
        #    
        #    yield (dense, l_label[iJet])
        
        for idx in a_shuffledIdx :
            
            arr = input_array[idx].todense()
            label = l_label[idx]
            
            #print("arr.dtype:", arr.dtype)
            
            yield (arr, label)
            
            #print('Generator yielded batch %d.' %(idx))
    
    
    def generator_1(idx) :
        
        yield (input_array[idx].todense(), l_label[idx])
    
    
    def generator_2() :
        
        for a_idx_batch in a_shuffledIdx_batch :
            
            input_array_batch = numpy.zeros((len(a_idx_batch),)+img_shape)
            
            #print("In generator_2(): starting pool.")
            pool = multiprocessing.Pool(processes = 20)#, maxtasksperchild = 1)
            #print("In generator_2(): started pool.")
            
            l_job = []
            
            for idx in a_idx_batch :
                
                l_job.append(pool.apply_async(input_array[idx].todense, ()))
                
            pool.close()
            
            #print("In generator_2(): submitted jobs.")
            
            l_isJobDone = [False] * len(l_job)
            
            while(False in l_isJobDone) :
                
                for iJob, job in enumerate(l_job) :
                    
                    #print iJob, job
                    
                    if (job is None) :
                        
                        continue
                    
                    if (not l_isJobDone[iJob] and job.ready()) :
                        
                        l_isJobDone[iJob] = True
                        
                        retVal = job.get()
                        #print("In generator_2(): ", iJob, retVal.shape, numpy.sum(retVal))
                        
                        input_array_batch[iJob] = retVal
            
            pool.join()
            
            #print("In generator_2(): closed and joined pool.")
            #print("In generator_2(): ", input_array_batch, numpy.sum(input_array_batch))
            #print("In generator_2(): ", a_idx_batch)
            
            yield (input_array_batch, a_idx_batch)
    
    
    dummy_img = numpy.empty(img_shape, dtype = numpy.float16)
    
    #sess = tensorflow.compat.v1.Session()
    
    @tensorflow.function
    def convert_idx(idx) :
        
        print("*"*10, type(idx), idx, str(idx))#, str(idx.op))
        print("*"*10, tensorflow.executing_eagerly(), tensorflow.compat.v1.executing_eagerly())
        
        #with tensorflow.compat.v1.Session().as_default() :
        #    print(idx.eval())
        
        print(idx.numpy())
        
        idx = idx.numpy()
        
        #print(idx+1, idx+2, idx+3)
        #
        #with tensorflow.compat.v1.Session() as sess :
        #    print(sess.run(idx))
        
        #if ("args_" in str(idx)) :
        #    
        #    return (dummy_img, -1)
        
        #print(sess.run(idx))
        
        #return (input_array[idx].todense(), l_label[idx])
        
        return idx
    
    @tensorflow.function
    def get_entry_and_label(idx) :
        
        #print("%"*10, type(idx), idx, str(idx))#, str(idx.op))
        #print("%"*10, tensorflow.executing_eagerly(), tensorflow.compat.v1.executing_eagerly())
        
        return (input_array[idx].todense(), l_label[idx])
    
    
    
    #dataset_image = tensorflow.data.Dataset.from_tensor_slices(input_array)
    #dataset_image = tensorflow.data.Dataset.from_generator(generator, output_types = (tensorflow.float16), output_shapes = (tensorflow.TensorShape(img_shape)))
    #dataset_image = tensorflow.data.Dataset.from_generator(generator, output_signature = tensorflow.TensorSpec(shape = (50, 50, 8), dtype = tensorflow.float16))
    #dataset_image = tensorflow.data.Dataset.from_generator(generator, output_types = (tensorflow.float16, l_label.dtype), output_shapes = (img_shape, ()))
    
    #dataset_label = tensorflow.data.Dataset.from_tensor_slices(l_label)
    #print(input_dataset.element_spec)
    
    print("=====> Creating dataset... Memory:", getMemoryMB())
    
    #dataset_train = tensorflow.data.Dataset.zip((dataset_image, dataset_label)).shuffle(buffer_size = nJetTot).batch(batch_size)
    #dataset_train = tensorflow.data.Dataset.from_generator(generator, output_types = (tensorflow.float16, l_label.dtype), output_shapes = (img_shape, ())).shuffle(buffer_size = nJetTot, reshuffle_each_iteration = False).batch(batch_size)
    
    #dataset_train = tensorflow.data.Dataset.from_generator(generator, output_types = (tensorflow.float16, l_label.dtype), output_shapes = (img_shape, ())).batch(batch_size).prefetch(tensorflow.data.AUTOTUNE)
    #dataset_train = tensorflow.data.Dataset.from_generator(generator, output_types = (tensorflow.float16, l_label.dtype), output_shapes = (img_shape, ())).batch(batch_size).prefetch(1000)
    #dataset_train = tensorflow.data.Dataset.from_generator(generator, output_types = (tensorflow.float16, l_label.dtype), output_shapes = (img_shape, ())).prefetch(5*batch_size).batch(batch_size)
    
    #dataset_train = tensorflow.data.Dataset.from_generator(generator, output_types = (tensorflow.float16, l_label.dtype), output_shapes = (img_shape, ())).batch(batch_size, num_parallel_calls = 100).prefetch(100)
    
    #dataset_train = tensorflow.data.Dataset.from_tensor_slices(a_shuffledIdx).map(mapped_func, num_parallel_calls = tensorflow.data.AUTOTUNE).batch(batch_size = batch_size, num_parallel_calls = tensorflow.data.AUTOTUNE).prefetch(100)
    
    #dataset_train = tensorflow.data.Dataset.from_tensor_slices(a_shuffledIdx).map(
    #    lambda idx : tensorflow.py_function(
    #        convert_idx,
    #        [idx],
    #        l_label.dtype,
    #    ),
    #    num_parallel_calls = tensorflow.data.AUTOTUNE,
    #).map(
    #    get_entry_and_label,
    #    num_parallel_calls = tensorflow.data.AUTOTUNE,
    #).batch(
    #    batch_size = batch_size,
    #    num_parallel_calls = tensorflow.data.AUTOTUNE,
    #).prefetch(100)
    
    #dataset_train = tensorflow.data.Dataset.from_tensor_slices(a_shuffledIdx).map(
    #    lambda idx : tensorflow.numpy_function(
    #        get_entry_and_label,
    #        [idx],
    #        (tensorflow.float16, l_label.dtype),
    #    ),
    #    num_parallel_calls = tensorflow.data.AUTOTUNE,
    #).batch(
    #    batch_size = batch_size,
    #    num_parallel_calls = tensorflow.data.AUTOTUNE,
    #).prefetch(100)
    
    
    #dataset_train = tensorflow.data.Dataset.from_tensor_slices(a_shuffledIdx).map(lambda idx : py_func(mapped_func, [idx], output_types = (tensorflow.float16, l_label.dtype), output_shapes = (img_shape, ())), num_parallel_calls = nCpu).batch(batch_size = batch_size, num_parallel_calls = tensorflow.data.AUTOTUNE).prefetch(100)
    
    dataset_image = tensorflow.data.Dataset.from_tensor_slices(input_array)
    dataset_label = tensorflow.data.Dataset.from_tensor_slices(l_label)
    dataset_train = tensorflow.data.Dataset.zip((dataset_image, dataset_label)).shuffle(buffer_size = len(l_label), reshuffle_each_iteration = False).batch(batch_size = batch_size, num_parallel_calls = tensorflow.data.AUTOTUNE).prefetch(tensorflow.data.AUTOTUNE)
    
    #dataset_train = tensorflow.data.Dataset.from_tensor_slices(a_shuffledIdx).interleave(
    #    lambda idx : tensorflow.data.Dataset.from_generator(
    #        generator_1,
    #        args = (idx,),
    #        output_types = (tensorflow.float16, l_label.dtype),
    #        output_shapes = (img_shape, ()),
    #    ),
    #    num_parallel_calls = int(nCpu/2),
    #    #cycle_length = nCpu,
    #).batch(
    #    batch_size = batch_size,
    #    num_parallel_calls = int(nCpu/2),
    #)#.prefetch(50)
    
    
    #dataset_train = tensorflow.data.Dataset.from_tensor_slices(a_shuffledIdx).interleave(
    #    lambda idx : get_entry_and_label(idx),
    #    num_parallel_calls = nCpu,
    #    #cycle_length = nCpu,
    #).batch(
    #    batch_size = batch_size,
    #    num_parallel_calls = nCpu,
    #).prefetch(50)
    
    
    #dataset_train = tensorflow.data.Dataset.from_generator(generator_2, output_types = (tensorflow.float16, l_label.dtype), output_shapes = ((None,)+img_shape, (None,))).prefetch(2)
    
    
    print("=====> Created dataset... Memory:", getMemoryMB())
    print(dataset_train.element_spec)
    
    
    ##### Dataset debug plot #####
    
    #print("="*10)
    #
    #l_imgArr = []
    #
    #l_count = []
    #
    #for (iCat, cat) in enumerate(d_catInfo.keys()) :
    #    
    #    l_imgArr.append([])
    #    l_count.append(0)
    #    
    #    for iLayer in range(0, nLayer) :
    #        
    #        l_imgArr[iCat].append(numpy.zeros((img_shape[0], img_shape[1])))
    #
    #
    #for element in dataset_train.as_numpy_iterator():
    #    
    #    ##print(element.shape)
    #    #print(type(element[0]), element[0].shape)
    #    #print(type(element[1]), element[1])
    #    
    #    #catNum = element[1]
    #    #layerNum = element[0][-1]
    #    
    #    for (iCat, cat) in enumerate(d_catInfo.keys()) :
    #        
    #        l_catIdx = numpy.nonzero(element[1] == iCat)[0]
    #        
    #        l_count[iCat] += len(l_catIdx)
    #        
    #        for iLayer in range(0, nLayer) :
    #            
    #            l_imgArr[iCat][iLayer] += element[0][l_catIdx, :, :, iLayer].sum(axis = 0)
    #
    #
    #
    #for cat in d_catInfo.keys() :
    #    
    #    print("cat.nJet, l_count[iCat]:", d_catInfo[cat].nJet, l_count[cat])
    #    
    #    for iLayer in range(0, nLayer) :
    #        
    #        arr_img = l_imgArr[cat][iLayer]
    #        arr_img /= float(l_count[cat])
    #        
    #        fig = matplotlib.pyplot.figure(figsize = [10, 8])
    #        axis = fig.add_subplot(1, 1, 1)
    #        axis.set_aspect("equal", "box")
    #        img = axis.imshow(
    #            arr_img,
    #            norm = matplotlib.colors.LogNorm(vmin = 1e-6, vmax = 1.0),
    #            origin = "lower",
    #            cmap = matplotlib.cm.get_cmap("nipy_spectral"),
    #        )
    #        
    #        fig.suptitle("Category %d, Layer %d" %(cat, iLayer))
    #        
    #        fig.colorbar(img, ax = axis)
    #        
    #        fig.tight_layout()
    #        
    #        fig.savefig("jetImage_cat%d_layer%d_fromTensorFlowDataset.pdf" %(cat, iLayer))
    
    
    ##exit()
    
    
    model = networks.d_network["CNN1"]
    model.summary()
    
    print("=====> Compiling model... Memory:", getMemoryMB())
    
    model.compile(
        optimizer = "adam",
        #loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = False), # no need to turn on from_logits if the network output is already a probability distribution like softmax
        metrics = ["accuracy"],
        run_eagerly = True,
    )
    
    print("=====> Compiled model... Memory:", getMemoryMB())
    
    
    base_result_dir = "training_results"
    
    out_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    checkpoint_file = "%s/model_checkpoints/%s/weights_{epoch:d}.hdf5" %(base_result_dir, out_tag)
    
    checkpoint_callback  =  tensorflow.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_file,
        monitor = "val_loss",
        verbose = 0,
        save_best_only = False,
        save_weights_only = False,
        mode = "auto",
        save_freq = "epoch",
        #options = None,
        #**kwargs
    )
    
    
    tensorboard_dir = "%s/tensorboard/%s" %(base_result_dir, out_tag)
    
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
        log_dir = tensorboard_dir,
        histogram_freq = 1,
        write_graph = True,
        write_images = True,
        write_steps_per_second = False,
        update_freq = "epoch",
        profile_batch = (1, nBatch),
        embeddings_freq = 0,
        embeddings_metadata = None,
    )
    
    print("=====> Starting fit... Memory:", getMemoryMB())
    
    history = model.fit(
        x = dataset_train,
        epochs = 20,
        #batch_size = batch_size,
        #validation_data = dataset_train,
        shuffle = False,
        #max_queue_size = 20,
        #use_multiprocessing = True,
        #workers = nCpu_use,
        callbacks = [checkpoint_callback, tensorboard_callback]
    )



if (__name__ == "__main__") :
    
    main()
