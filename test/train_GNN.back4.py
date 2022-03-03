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
import tensorflow
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

from tf_keras_model import get_particle_net, get_particle_net_lite


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


def loadBatchSlice(
    sliceIdx_global : int,
    sliceIdx_batch : int,
    d_data_sharedBuf_name : dict,
    d_data_shape : dict,
    d_branchName_alias : dict,
    catNum : int,
    sampleIdx : int,
    fileIdx : int,
    evtIdx_start : int,
    jetIdx_start : int,
    evtIdx_end : int,
    jetIdx_end : int,
    maxParticles : int,
    sliceSize : int,
    fileAndTreeName : str,
    cut : str,
    debugStr : str = "",
    printEvery : int = 1000,
) :
    
    d_data = sortedcontainers.SortedDict()
    d_data_sharedBuf = sortedcontainers.SortedDict()
    
    for key in list(d_data_sharedBuf_name.keys()) :
        
        d_data_sharedBuf[key] = multiprocessing.shared_memory.SharedMemory(name = d_data_sharedBuf_name[key])
        d_data[key] = numpy.ndarray(d_data_shape[key], buffer = d_data_sharedBuf[key].buf)
    
    
    #catNum = d_sliceInfo["a_catNum"][sliceIdx_global]
    #
    #sampleIdx = d_sliceInfo["a_sampleIdx"][sliceIdx_global]
    #fileIdx = d_sliceInfo["a_fileIdx"][sliceIdx_global]
    #
    #evtIdx_start = d_sliceInfo["a_evtIdx_start"][sliceIdx_global]
    #jetIdx_start = d_sliceInfo["a_jetIdx_start"][sliceIdx_global]
    #
    #evtIdx_end = d_sliceInfo["a_evtIdx_end"][sliceIdx_global]
    #jetIdx_end = d_sliceInfo["a_jetIdx_end"][sliceIdx_global]
    
    #sample_fileAndTreeName = catInfo.l_sample_fileAndTreeName[sampleIdx]
    #fileAndTreeName = sample_fileAndTreeName[fileIdx]
    
    if (len(debugStr)) :
        
        print("Starting:", debugStr)
    
    #print("Reading jets from: %s" %(fileAndTreeName))
    
    with uproot.open(fileAndTreeName) as tree :
        
        for dataKey in list(d_branchName_alias.keys()) :
            
            branches = tree.arrays(
                expressions = list(d_branchName_alias[dataKey].keys()),
                cut = cut,
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
                
                
                a_br_padded = awkward.pad_none(a_br, target = maxParticles, axis = 1, clip = True)
                a_br_masked = awkward.fill_none(a_br_padded, 0)
                
                i1 = sliceIdx_batch*sliceSize
                i2 = i1 + sliceSize
                
                d_data[dataKey][i1: i2, :, iBrKey] = a_br_masked
    
    
    for key in list(d_data_sharedBuf_name.keys()) :
        
        d_data_sharedBuf[key].close()
    
    if (len(debugStr)) :
        
        print("Processed:", debugStr)
    
    return 0


@dataclasses.dataclass
class MyDataset :
    
    d_loadConfig        : Dict
    d_catInfo           : Dict
    d_sliceInfo         : Dict
    a_sliceIdx_batched  : List
    sliceSize           : int
    batchSize           : int
    #nBatch              : int
    
    def __post_init__(
        self,
    ) :
        
        self.nCategory = len(self.d_loadConfig["categories"])
        self.nBatch = len(self.a_sliceIdx_batched)
        
        self.d_data_shape = sortedcontainers.SortedDict()
        self.d_data_inputShape = sortedcontainers.SortedDict()
        self.d_data_outputShape = sortedcontainers.SortedDict()
        
        for key in ["points", "features", "mask"] :
            
            l_temp = self.d_loadConfig[key]
            
            if type(l_temp) not in [list, tuple] :
                
                l_temp = [l_temp]
            
            shape = (self.batchSize, self.d_loadConfig["maxParticles"], len(l_temp))
            self.d_data_shape[key] = shape
            
            inputShape = (self.d_loadConfig["maxParticles"], len(l_temp))
            self.d_data_inputShape[key] = inputShape
            
            outputShape = (None, self.d_loadConfig["maxParticles"], len(l_temp))
            self.d_data_outputShape[key] = outputShape
            #self.d_data_outputShape[key] = shape
    
    
    def generateBatch(
        self,
    ) :
        
        for batchIdx in range(0, self.nBatch) :
            
            d_data = sortedcontainers.SortedDict()
            d_data_sharedBuf = sortedcontainers.SortedDict()
            d_data_sharedBuf_name = sortedcontainers.SortedDict()
            #d_data_shape = sortedcontainers.SortedDict()
            
            l_catNum = []
            l_sliceSize = []
            
            for key in ["points", "features", "mask"] :
                
                l_temp = self.d_loadConfig[key]
                
                if type(l_temp) not in [list, tuple] :
                    
                    l_temp = [l_temp]
                
                shape = (self.batchSize, self.d_loadConfig["maxParticles"], len(l_temp))
                a_temp = numpy.empty(shape)
                
                #self.d_data_shape[key] = shape
                d_data_sharedBuf[key] = multiprocessing.shared_memory.SharedMemory(create = True, size = a_temp.nbytes)
                d_data[key] = numpy.ndarray(a_temp.shape, buffer = d_data_sharedBuf[key].buf)
                d_data_sharedBuf_name[key] = d_data_sharedBuf[key].name
            
            d_branchName_alias = sortedcontainers.SortedDict()
            d_branchName_alias["points"]    = sortedcontainers.SortedDict({"coor%d" %(_i): _expr for _i, _expr in enumerate(self.d_loadConfig["points"])})
            d_branchName_alias["features"]  = sortedcontainers.SortedDict({"feat%d" %(_i): _expr for _i, _expr in enumerate(self.d_loadConfig["features"])})
            d_branchName_alias["mask"]      = sortedcontainers.SortedDict({"mask": self.d_loadConfig["mask"]})
            
            nCpu = multiprocessing.cpu_count()
            nCpu_frac = self.d_loadConfig["cpuFrac"]
            nCpu_use = int(nCpu_frac*nCpu)
            
            pool = multiprocessing.Pool(processes = nCpu_use, maxtasksperchild = 1)
            l_job = []
            
            a_sliceIdx = self.a_sliceIdx_batched[batchIdx]
            
            for sliceIdx_batch, sliceIdx_global in enumerate(a_sliceIdx) :
                
                catNum = self.d_sliceInfo["a_catNum"][sliceIdx_global]
                
                sampleIdx = self.d_sliceInfo["a_sampleIdx"][sliceIdx_global]
                fileIdx = self.d_sliceInfo["a_fileIdx"][sliceIdx_global]
                
                evtIdx_start = self.d_sliceInfo["a_evtIdx_start"][sliceIdx_global]
                jetIdx_start = self.d_sliceInfo["a_jetIdx_start"][sliceIdx_global]
                
                evtIdx_end = self.d_sliceInfo["a_evtIdx_end"][sliceIdx_global]
                jetIdx_end = self.d_sliceInfo["a_jetIdx_end"][sliceIdx_global]
                
                catInfo = self.d_catInfo[catNum]
                
                sample_fileAndTreeName = catInfo.l_sample_fileAndTreeName[sampleIdx]
                fileAndTreeName = sample_fileAndTreeName[fileIdx]
                
                l_catNum.append(catNum)
                l_sliceSize.append(self.sliceSize)
                
                sliceDebugStr = (
                    "batch idx %d (total %d), "
                    "batch slice idx %d (total %d), "
                    #"global slice idx %d (total %d), "
                    "category %d (%s), "
                    "sample %d, "
                    "file %d, "
                    %(
                    batchIdx, self.nBatch,
                    sliceIdx_batch, len(a_sliceIdx),
                    #sliceIdx_global, nSlice,
                    catNum, catInfo.catName,
                    sampleIdx,
                    fileIdx,
                ))
                
                l_job.append(pool.apply_async(
                    loadBatchSlice,
                    (),
                    dict(
                        sliceIdx_global = sliceIdx_global,
                        sliceIdx_batch = sliceIdx_batch,
                        d_data_sharedBuf_name = d_data_sharedBuf_name,
                        d_data_shape = self.d_data_shape,
                        d_branchName_alias = d_branchName_alias,
                        catNum = catNum,
                        sampleIdx = sampleIdx,
                        fileIdx = fileIdx,
                        evtIdx_start = evtIdx_start,
                        jetIdx_start = jetIdx_start,
                        evtIdx_end = evtIdx_end,
                        jetIdx_end = jetIdx_end,
                        maxParticles = self.d_loadConfig["maxParticles"],
                        sliceSize = self.sliceSize,
                        fileAndTreeName = fileAndTreeName,
                        cut = catInfo.cut,
                        debugStr = "",
                        #debugStr = sliceDebugStr,
                        #printEvery = 1000,
                    ),
                    #callback = inpData.fill,
                ))
                
                
                
                #print("d_data:")
                #pprint.pprint(d_data, depth = 5)
            
            
            pool.close()
            
            l_isJobDone = [False] * len(l_job)
            
            while(False in l_isJobDone) :
                
                for iJob, job in enumerate(l_job) :
                    
                    if (job is None) :
                        
                        continue
                    
                    if (not l_isJobDone[iJob] and job.ready()) :
                        
                        l_isJobDone[iJob] = True
                        
                        retVal = job.get()
                        #print("retVal", retVal)
                        
                        l_job[iJob] = None
                        
                        if (not sum(l_isJobDone) % 10) :
                            
                            gc.collect()
                        
                        #print("[batch idx %d (num %d/%d)] processed output of read-job num %d (%d/%d done)." %(batchIdx, batchIdx+1, self.nBatch, iJob+1, sum(l_isJobDone), len(l_job)))
                        #print("=====> Memory:", utils.getMemoryMB())
            
            pool.join()
            
            #pprint.pprint(d_data)
            #print(d_data["mask"].shape)
            
            d_data_cpy = sortedcontainers.SortedDict()
            
            for key in ["points", "features", "mask"] :
                
                d_data_cpy[key] = d_data[key].copy()
                
                d_data_sharedBuf[key].close()
                d_data_sharedBuf[key].unlink()
            
            #print("Adding result of batch idx %d to queue." %(batchIdx))
            #print("Yielding batch idx %d (%d/%d)." %(batchIdx, batchIdx+1, self.nBatch))
            
            gc.collect()
            #queue.put(d_data_cpy)
            
            #a_label = numpy.repeat(l_catNum, l_sliceSize)
            a_label = tensorflow.one_hot(indices = numpy.repeat(l_catNum, l_sliceSize), depth = self.nCategory)
            
            yield((d_data_cpy, a_label))
            
            #queue.put((d_data_sharedBuf_name, self.d_data_shape))
            #queue.put((d_data_sharedBuf, self.d_data_shape))


def benchmark(dataset, num_epochs = 5):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        print("Training epoch %d/%d" %(epoch_num+1, num_epochs))
        for sample in dataset:
            
            x, y = sample
            print({key: x[key].get_shape() for key in x.keys()}, y.get_shape())
            # Performing a training step
            time.sleep(0.1)
    print("Execution time:", time.perf_counter() - start_time)




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
    
    nCategory = len(d_loadConfig["categories"])
    
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
    
    
    print("nSlice", nSlice)
    print("sliceSize", sliceSize)
    print("nBatch", nBatch)
    print("batchSize", batchSize)
    print("nSlicePerBatch", nSlicePerBatch)
    
    
    # Shuffle slices
    a_sliceIdx_shf = numpy.arange(0, nSlice)
    numpy.random.shuffle(a_sliceIdx_shf)
    
    a_sliceIdx_shf_batched = numpy.split(a_sliceIdx_shf, nBatch)
    
    #for key in list(d_sliceInfo.keys()) :
    #    
    #    d_sliceInfo[key] = operator.itemgetter(*l_sliceIdx_shf)(d_sliceInfo[key])
    
    
    mydataset_trn = MyDataset(
        d_loadConfig = d_loadConfig,
        d_catInfo = d_catInfo_trn,
        d_sliceInfo = d_sliceInfo,
        a_sliceIdx_batched = a_sliceIdx_shf_batched,
        sliceSize = sliceSize,
        batchSize = batchSize,
        #nBatch = nBatch,
    )
    
    
    dataset_trn = tensorflow.data.Dataset.from_generator(
        mydataset_trn.generateBatch,
        output_types=({_: float for _ in mydataset_trn.d_data_outputShape}, tensorflow.int64),
        #output_shapes = (mydataset_trn.d_data_outputShape, (batchSize,)),
        output_shapes = (mydataset_trn.d_data_outputShape, (None, nCategory)),
    ).prefetch(tensorflow.data.AUTOTUNE)
    
    #dataset = tensorflow.data.Dataset.range(nBatch).interleave(
    #    lambda _idx: tensorflow.data.Dataset.from_generator(
    #        dataset_trn.generateBatch,
    #        output_types={_: float for _ in dataset_trn.d_data_outputShape},
    #        output_shapes = dataset_trn.d_data_outputShape,
    #    ),
    #    
    #    cycle_length = 30,
    #    num_parallel_calls = tensorflow.data.AUTOTUNE,
    #    deterministic = False,
    #)#.prefetch(20)
    
    #benchmark(dataset_trn)
    
    
    #model = get_particle_net(nCategory, input_shapes = mydataset_trn.d_data_inputShape)
    model = get_particle_net_lite(nCategory, input_shapes = mydataset_trn.d_data_inputShape)
    
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 10:
            lr *= 0.1
        elif epoch > 20:
            lr *= 0.01
        #logging.info("Learning rate: %f"%lr)
        return lr
    
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = lr_schedule(0)),
        metrics=["accuracy"],
    )
    
    model.summary()
    
    
    model.fit(
        x = dataset_trn,
        epochs = 5, # --- train only for 1 epoch here for demonstration ---
        #validation_data = (val_dataset.X, val_dataset.y),
        shuffle = False,
        #callbacks=callbacks
    )
    
    
    
    #print("counts", l)
    print("nSlice", nSlice)
    print("sliceSize", sliceSize)
    print("nBatch", nBatch)
    print("batchSize", batchSize)
    print("nSlicePerBatch", nSlicePerBatch)
    
    
    return 0


if (__name__ == "__main__") :
    
    main()
