from __future__ import print_function

#import mxnet
import argparse
import awkward
import collections
#import cppyy
#import cppyy.ll
import concurrent.futures
import dataclasses
import datetime
import gc
#import keras
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import memory_profiler
import multiprocessing
import multiprocessing.managers
import multiprocessing.shared_memory
import numpy
import operator
import os
import PIL
import pprint
import psutil
import pympler
import sklearn
import sklearn.metrics
import sortedcontainers
import sparse
import sys
import tabulate
#import tensorflow
import time
import uproot
import yaml

from typing import List, Set, Dict, Tuple, Optional

#from tensorflow.keras import datasets, layers, models
#from tensorflow.keras import mixed_precision

#policy = mixed_precision.Policy("mixed_float16")
#mixed_precision.set_global_policy(policy)

#import networks
import utils


@dataclasses.dataclass
class CategoryInfo :
    
    catNum:         int
    catName:        str
    samples:        List[str]
    cut:            str
    
    def __post_init__(
        self,
    ) :
        
        self.nJet = 0
        
        self.l_sample_fileAndTreeName = [None] * len(self.samples)
        self.l_sample_weight = [None] * len(self.samples)
        self.l_sample_nJet = [0] * len(self.samples)
        self.l_sample_nJetMax = [0] * len(self.samples)
        
        for iSample, sample in enumerate(self.samples) :
            
            l_token = sample.strip().split(":")
            
            fName = l_token[0]
            treeName = l_token[1]
            weight = float(l_token[2])
            nJetMax = int(l_token[3])
            
            self.l_sample_fileAndTreeName[iSample] = ["%s:%s" %(ele, treeName) for ele in numpy.loadtxt(fName, dtype = str, delimiter = "*"*100)]
            self.l_sample_weight[iSample] = weight
            self.l_sample_nJetMax[iSample] = nJetMax
        
        ## Including the stop index (i.e. start --> stop)
        #self.l_sample_treeIdx_start = [0] * len(self.samples)
        #self.l_sample_treeIdx_stop = [-1] * len(self.samples)
        #
        ## Excluding the stop index (i.e. start --> stop-1)
        #self.l_sample_treeEventSliceIdx_start = [[0] * len(ele) for ele in self.l_sample_fileAndTreeName]
        #self.l_sample_treeEventSliceIdx_stop = [[-1] * len(ele) for ele in self.l_sample_fileAndTreeName]


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
    
    batchSize = d_loadConfig["batchSize"]
    sliceSize = d_loadConfig["sliceSize"]
    
    assert(batchSize%sliceSize == 0)
    
    nSlicePerBatch = int(batchSize/sliceSize)
    
    d_catInfo_trn = sortedcontainers.SortedDict()
    
    for iCat, cat in enumerate(d_loadConfig["categories"]) :
        
        d_catInfo_trn[iCat] = CategoryInfo(
            catNum = iCat,
            catName = cat["name"],
            samples = cat["samples"],
            cut = cat["cut"],
        )
    
    
    d_sliceInfo = {
        "a_catNum":             [],
        
        "a_sampleIdx":          [],
        "a_fileIdx":            [],
        
        "a_evtIdx_start":       [],
        "a_jetIdx_start":       [],
        
        "a_evtIdx_end":         [],
        "a_jetIdx_end":         [],
    }
    
    
    for (iCat, cat) in enumerate(d_catInfo_trn.keys()) :
        
        catInfo_trn = d_catInfo_trn[cat]
        #catInfo_tst = d_catInfo_tst[cat]
        
        for iSample, sample_fileAndTreeName in enumerate(catInfo_trn.l_sample_fileAndTreeName):
            
            nJet_sample = 0
            
            for (iTree, fileAndTreeName) in enumerate(sample_fileAndTreeName[0: 2]) :
                
                print("Counting jets in: %s" %(fileAndTreeName))
                
                with uproot.open(fileAndTreeName) as tree :
                    
                    brName = "cut"
                    
                    branches = tree.arrays(
                        expressions = [brName],
                        cut = catInfo_trn.cut,
                        #entry_start = entry_start,
                        #entry_stop = entry_stop,
                        aliases = {brName: catInfo_trn.cut},
                        #language = utils.uproot_lang,
                    )
                    
                    a_cut = branches[brName]
                    a_jetCount = awkward.count(a_cut, axis = 1)
                    a_jetCount_cumSum = numpy.cumsum(awkward.flatten(a_cut).to_numpy())
                    a_jetCount_cumSum_shaped = awkward.unflatten(a_jetCount_cumSum, a_jetCount)
                    
                    print(tree.num_entries, len(a_cut))
                    print("a_cut", a_cut)
                    print("a_jetCount", a_jetCount)
                    print("a_jetCount_cumSum", a_jetCount_cumSum)
                    
                    a_localIdx = awkward.local_index(a_cut)
                    
                    a_evtIdx_start = awkward.flatten(awkward.where(a_jetCount_cumSum_shaped%sliceSize == 1, range(0, len(a_cut)), -1))
                    a_evtIdx_start = a_evtIdx_start[awkward.where(a_evtIdx_start >= 0)]
                    
                    a_jetIdx_start = awkward.flatten(awkward.where(a_jetCount_cumSum_shaped%sliceSize == 1, a_localIdx, -1))
                    a_jetIdx_start = a_jetIdx_start[awkward.where(a_jetIdx_start >= 0)]
                    
                    
                    a_evtIdx_end = awkward.flatten(awkward.where(a_jetCount_cumSum_shaped%sliceSize == 0, range(0, len(a_cut)), -1))
                    a_evtIdx_end = a_evtIdx_end[awkward.where(a_evtIdx_end >= 0)]
                    
                    a_jetIdx_end = awkward.flatten(awkward.where(a_jetCount_cumSum_shaped%sliceSize == 0, a_localIdx, -1))
                    a_jetIdx_end = a_jetIdx_end[awkward.where(a_jetIdx_end >= 0)]
                    
                    
                    print("a_evtIdx_start", len(a_evtIdx_start), a_evtIdx_start)
                    print("a_jetIdx_start", len(a_jetIdx_start), a_jetIdx_start)
                    print("a_evtIdx_end", len(a_evtIdx_end), a_evtIdx_end)
                    print("a_jetIdx_end", len(a_jetIdx_end), a_jetIdx_end)
                    
                    max_slices = min(len(a_evtIdx_start), len(a_evtIdx_end))
                    
                    a_evtIdx_start = a_evtIdx_start[: max_slices]
                    a_jetIdx_start = a_jetIdx_start[: max_slices]
                    
                    a_evtIdx_end = a_evtIdx_end[: max_slices]
                    a_jetIdx_end = a_jetIdx_end[: max_slices]
                    
                    a_sampleIdx = [iSample] * max_slices
                    a_fileIdx = [iTree] * max_slices
                    
                    a_catNum = [catInfo_trn.catNum] * max_slices
                    
                    l = []
                    
                    for i in range(0, len(a_evtIdx_start)) :
                        
                        evtIdx_start = a_evtIdx_start[i]
                        jetIdx_start = a_jetIdx_start[i]
                        
                        evtIdx_end = a_evtIdx_end[i]
                        jetIdx_end = a_jetIdx_end[i]
                        
                        a_tmp = awkward.flatten(a_cut[evtIdx_start: evtIdx_end+1])
                        a_tmp = a_tmp[jetIdx_start: ]
                        
                        d = len(a_cut[evtIdx_end])
                        
                        a_tmp = a_tmp[: len(a_tmp)-d+jetIdx_end+1]
                        
                        l.append(awkward.count(a_tmp))
                    
                    
                    d_sliceInfo["a_catNum"].extend(a_catNum)
                    
                    d_sliceInfo["a_sampleIdx"].extend(a_sampleIdx)
                    d_sliceInfo["a_fileIdx"].extend(a_fileIdx)
                    
                    d_sliceInfo["a_evtIdx_start"].extend(a_evtIdx_start)
                    d_sliceInfo["a_jetIdx_start"].extend(a_jetIdx_start)
                    
                    d_sliceInfo["a_evtIdx_end"].extend(a_evtIdx_end)
                    d_sliceInfo["a_jetIdx_end"].extend(a_jetIdx_end)
                    
                    print("counts", l)
    
    
    
    
    
    # Choose nSlice slices such that nSlice*sliceSize is a multiple of batchSize
    nSlice = len(d_sliceInfo["a_sampleIdx"])
    
    #print(nSlice, sliceSize, batchSize)
    #assert((nSlice*sliceSize) % batchSize == 0)
    nBatch = int(nSlice/nSlicePerBatch)
    
    nSlice = nSlicePerBatch * nBatch
    
    
    # Shuffle slices
    a_sliceIdx_shf = numpy.arange(0, nSlice)
    numpy.random.shuffle(a_sliceIdx_shf)
    
    a_sliceIdx_shf_batched = numpy.split(a_sliceIdx_shf, nBatch)
    
    #for key in list(d_sliceInfo.keys()) :
    #    
    #    d_sliceInfo[key] = operator.itemgetter(*l_sliceIdx_shf)(d_sliceInfo[key])
    
    l = []
    
    for iBatch in range(0, nBatch) :
        
        a_sliceIdx = a_sliceIdx_shf_batched[iBatch]
        
        d_data = sortedcontainers.SortedDict()
        d_data["points"] =      numpy.zeros((batchSize, d_loadConfig["maxParticles"], len(d_loadConfig["points"])))
        d_data["features"] =    numpy.zeros((batchSize, d_loadConfig["maxParticles"], len(d_loadConfig["features"])))
        d_data["mask"] =        numpy.zeros((batchSize, d_loadConfig["maxParticles"], 1))
        
        d_branchName_alias = sortedcontainers.SortedDict()
        d_branchName_alias["points"] =      sortedcontainers.SortedDict({"coor%d" %(_i): _expr for _i, _expr in enumerate(d_loadConfig["points"])})
        d_branchName_alias["features"] =    sortedcontainers.SortedDict({"feat%d" %(_i): _expr for _i, _expr in enumerate(d_loadConfig["features"])})
        
        for iSlice, sliceIdx in enumerate(a_sliceIdx) :
            
            catNum = d_sliceInfo["a_catNum"][sliceIdx]
            
            sampleIdx = d_sliceInfo["a_sampleIdx"][sliceIdx]
            fileIdx = d_sliceInfo["a_fileIdx"][sliceIdx]
            
            evtIdx_start = d_sliceInfo["a_evtIdx_start"][sliceIdx]
            jetIdx_start = d_sliceInfo["a_jetIdx_start"][sliceIdx]
            
            evtIdx_end = d_sliceInfo["a_evtIdx_end"][sliceIdx]
            jetIdx_end = d_sliceInfo["a_jetIdx_end"][sliceIdx]
            
            catInfo_trn = d_catInfo_trn[catNum]
            
            print(
                "batch idx %d (total %d), "
                "batch slice idx %d (total %d), "
                "global slice idx %d (total %d), "
                "category %d (%s), "
                "sample %d, "
                "file %d, "
                %(
                iBatch, nBatch,
                iSlice, len(a_sliceIdx),
                sliceIdx, nSlice,
                catNum, catInfo_trn.catName,
                sampleIdx,
                fileIdx,
            ))
            
            sample_fileAndTreeName = catInfo_trn.l_sample_fileAndTreeName[sampleIdx]
            fileAndTreeName = sample_fileAndTreeName[fileIdx]
            
            print("Reading jets from: %s" %(fileAndTreeName))
            
            isMaskSet = False
            
            with uproot.open(fileAndTreeName) as tree :
                
                for dataKey in list(d_branchName_alias.keys()) :
                    
                    branches = tree.arrays(
                        expressions = list(d_branchName_alias[dataKey].keys()),
                        cut = catInfo_trn.cut,
                        entry_start = evtIdx_start,
                        entry_stop = evtIdx_end+1,
                        aliases = d_branchName_alias[dataKey],
                        language = utils.uproot_lang,
                    )
                    
                    #print(branches)
                    
                    for iBrKey, brKey in enumerate(list(d_branchName_alias[dataKey].keys())) :
                        
                        #print(brKey)
                        
                        a_br = awkward.flatten(branches[brKey], axis = 1)
                        c1 = len(a_br)
                        c2 = len(branches[brKey][-1])
                        a_br = a_br[jetIdx_start: c1-c2+jetIdx_end+1]
                        #a_br = a_br[: c1-c2+jetIdx_end+1]
                        l.append(len(a_br))
                        
                        
                        a_br_padded = awkward.pad_none(a_br, target = d_loadConfig["maxParticles"], axis = 1, clip = True)
                        a_br_masked = awkward.fill_none(a_br_padded, 0)
                        
                        i1 = iSlice*sliceSize
                        i2 = i1 + sliceSize
                        
                        d_data[dataKey][i1: i2, :, iBrKey] = a_br_masked
                        
                        if (not isMaskSet) :
                            
                            a_mask = ~awkward.is_none(a_br_padded, axis = 1)
                            d_data["mask"][i1: i2, :, 0] = a_mask
                            isMaskSet = True
                        
                        #print(a_mask.to_numpy())
                        #print(a_br_masked.to_numpy())
            
            #print("d_data:")
            #pprint.pprint(d_data, depth = 5)
            
    print("counts", l)
    
    
    return 0


if (__name__ == "__main__") :
    
    main()
