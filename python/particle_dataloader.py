from __future__ import print_function

import awkward
import collections
import copy
import concurrent.futures
import dataclasses
import datetime
import gc
import logging
import memory_profiler
import multiprocessing
import multiprocessing.managers
import multiprocessing.shared_memory
import numpy
import operator
import os
import pprint
import psutil
import pympler
import sortedcontainers
import sys
import time
import uproot

from typing import List, Set, Dict, Tuple, Optional

import utils


def loadSliceInfo(
    catNum,
    sampleIdx,
    catInfo,
    sliceSize,
) :
    l_fileAndTreeName = catInfo.l_sample_fileAndTreeName[sampleIdx]
    
    nJet_sample = 0
    nSlice_sample = 0
    
    d_sliceInfo_sample = copy.deepcopy(utils.d_sliceInfo_template)
    
    brk = False
    
    for (iTree, fileAndTreeName) in enumerate(l_fileAndTreeName) :
        
        print("Counting jets in: %s" %(fileAndTreeName))
        
        success = False
        
        while (not success and not brk) :
            
            try :
                
                with uproot.open(
                    fileAndTreeName,
                    #num_workers = nCpu_use,
                    #num_fallback_workers = nCpu_use,
                    #xrootd_handler = uproot.MultithreadedXRootDSource,
                    timeout = None,
                ) as tree :
                    
                    brName = "cut"
                    
                    branches = tree.arrays(
                        expressions = [brName],
                        cut = catInfo.cut,
                        aliases = {brName: catInfo.cut},
                        #language = utils.uproot_lang,
                    )
                    
                    a_cut = branches[brName]
                    a_jetCount = awkward.count(a_cut, axis = 1)
                    a_jetCount_cumSum = numpy.cumsum(awkward.flatten(a_cut).to_numpy())
                    a_jetCount_cumSum_shaped = awkward.unflatten(a_jetCount_cumSum, a_jetCount)
                    
                    print("tree.num_entries", tree.num_entries, ", len(a_cut)", len(a_cut))
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
                    
                    a_sampleIdx = [sampleIdx] * max_slices
                    a_fileIdx = [iTree] * max_slices
                    
                    a_catNum = [catInfo.catNum] * max_slices
                    
                    #l = []
                    
                    for idx in range(0, len(a_evtIdx_start)) :
                        
                        evtIdx_start = a_evtIdx_start[idx]
                        jetIdx_start = a_jetIdx_start[idx]
                        
                        evtIdx_end = a_evtIdx_end[idx]
                        jetIdx_end = a_jetIdx_end[idx]
                        
                        #a_tmp = awkward.flatten(a_cut[evtIdx_start: evtIdx_end+1])
                        #a_tmp = a_tmp[jetIdx_start: ]
                        #
                        #d = len(a_cut[evtIdx_end])
                        #
                        #a_tmp = a_tmp[: len(a_tmp)-d+jetIdx_end+1]
                        #
                        #l.append(awkward.count(a_tmp))
                    
                    
                    d_sliceInfo_sample["a_catNum"].extend(a_catNum)
                    
                    d_sliceInfo_sample["a_sampleIdx"].extend(a_sampleIdx)
                    d_sliceInfo_sample["a_fileIdx"].extend(a_fileIdx)
                    
                    d_sliceInfo_sample["a_evtIdx_start"].extend(a_evtIdx_start)
                    d_sliceInfo_sample["a_jetIdx_start"].extend(a_jetIdx_start)
                    
                    d_sliceInfo_sample["a_evtIdx_end"].extend(a_evtIdx_end)
                    d_sliceInfo_sample["a_jetIdx_end"].extend(a_jetIdx_end)
                    
                    nSlice_sample += len(a_evtIdx_start)
                    nJet_sample = nSlice_sample*sliceSize
                    
                    if (catInfo.l_sample_nJetMax[sampleIdx] > 0 and nJet_sample >= catInfo.l_sample_nJetMax[sampleIdx]) :
                        
                        success = True
                        brk = True
                        break
                    
                    #print("counts", l)
                
                success = True
            
            except Exception as exc:
                # Catch file read errors such as:
                # OSError: XRootD error: [FATAL] Auth failed: No protocols left to try
                
                success = False
                
                logging.warning("Exception while reading file: %s" %(fileAndTreeName))
                logging.warning(exc)
                logging.warning("Retrying...")
        
        
        if (brk) :
            
            break
    
    if (not tree.closed) :
        
        tree.close()
    
    
    d_result = {
        "catNum": catNum,
        "sampleIdx": sampleIdx,
        "nJet_sample": nJet_sample,
        "nSlice_sample": nSlice_sample,
        "d_sliceInfo_sample": d_sliceInfo_sample,
    }
    
    return d_result


def loadBatchSlice(
    sliceIdx_global         : int,
    sliceIdx_batch          : int,
    d_data_sharedBuf_name   : dict,
    d_data_shape            : dict,
    d_branchName_alias      : dict,
    catNum                  : int,
    sampleIdx               : int,
    fileIdx                 : int,
    evtIdx_start            : int,
    jetIdx_start            : int,
    evtIdx_end              : int,
    jetIdx_end              : int,
    maxParticles            : int,
    sliceSize               : int,
    fileAndTreeName         : str,
    cut                     : str,
    debugStr                : str       = "",
    printEvery              : int       = 1000,
) :
    
    d_data = sortedcontainers.SortedDict()
    d_data_sharedBuf = sortedcontainers.SortedDict()
    
    for key in list(d_data_sharedBuf_name.keys()) :
        
        d_data_sharedBuf[key] = multiprocessing.shared_memory.SharedMemory(name = d_data_sharedBuf_name[key])
        d_data[key] = numpy.ndarray(d_data_shape[key], buffer = d_data_sharedBuf[key].buf)
    
    
    if (len(debugStr)) :
        
        print("Starting:", debugStr)
    
    #print("Reading jets from: %s" %(fileAndTreeName))
    
    success = False
    
    while(not success) :
        
        try :
            
            with uproot.open(
                fileAndTreeName,
                #xrootd_handler = uproot.MultithreadedXRootDSource,
                timeout = None,
            ) as tree :
                
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
                        
                        
                        i1 = sliceIdx_batch*sliceSize
                        i2 = i1 + sliceSize
                        
                        if (len(d_data_shape[dataKey]) == 3) :
                            
                            a_br_padded = awkward.pad_none(a_br, target = maxParticles, axis = 1, clip = True)
                            a_br_masked = awkward.fill_none(a_br_padded, 0)
                            d_data[dataKey][i1: i2, :, iBrKey] = a_br_masked
                        
                        else :
                            
                            d_data[dataKey][i1: i2, iBrKey] = a_br
                        
                        #if (dataKey == "mask") :
                        #    
                        #    print("mask_sum", numpy.sum(a_br))
                        
                        success = True
        
        except Exception as exc:
            # Catch file read errors such as:
            # OSError: XRootD error: [FATAL] Auth failed: No protocols left to try
            
            success = False
            
            logging.warning("Exception while reading file: %s" %(fileAndTreeName))
            logging.warning(exc)
            logging.warning("Retrying after 5 seconds...")
            time.sleep(5)
            
            ##return 1
    
    if (not tree.closed) :
        
        tree.close()
    
    
    
    for key in list(d_data_sharedBuf_name.keys()) :
        
        d_data_sharedBuf[key].close()
    
    if (len(debugStr)) :
        
        print("Processed:", debugStr)
    
    return 0


@dataclasses.dataclass
class ParticleDataset :
    
    d_loadConfig                : Dict
    d_catInfo                   : Dict
    d_sliceInfo                 : Dict
    a_sliceIdx_batched          : List
    l_particleFeatureKey        : List
    jetFeatureKey               : str
    sliceSize                   : int
    batchSize                   : int
    shuffleBatchesEveryEpoch    : bool
    
    def __post_init__(
        self,
    ) :
        
        self.nCategory = len(self.d_loadConfig["categories"])
        self.nBatch = len(self.a_sliceIdx_batched)
        
        self.d_data_shape = sortedcontainers.SortedDict()
        self.d_data_inputShape = sortedcontainers.SortedDict()
        self.d_data_outputShape = sortedcontainers.SortedDict()
        
        for key in self.l_particleFeatureKey :
            
            l_temp = self.d_loadConfig[key]
            
            if type(l_temp) not in [list, tuple] :
                
                l_temp = [l_temp]
            
            shape = (self.batchSize, self.d_loadConfig["maxParticles"], len(l_temp))
            self.d_data_shape[key] = shape
            
            inputShape = (self.d_loadConfig["maxParticles"], len(l_temp))
            self.d_data_inputShape[key] = inputShape
            
            outputShape = (None, self.d_loadConfig["maxParticles"], len(l_temp))
            self.d_data_outputShape[key] = outputShape
            
        
        if (self.jetFeatureKey is not None) :
            
            l_temp = self.d_loadConfig[self.jetFeatureKey]
            
            self.d_data_shape[self.jetFeatureKey] = (self.batchSize, len(l_temp))
            self.d_data_inputShape[self.jetFeatureKey] = (len(l_temp),)
            self.d_data_outputShape[self.jetFeatureKey] = (None, len(l_temp))
            
        
        
        self.queueSize_max = min(20, self.nBatch)
        self.parallelBatches_max = min(10, self.nBatch)
        
        self.rng = numpy.random.default_rng()
        
    
    
    def countProcRunning(
        self
    ) :
        
        return sum([(not _proc._closed and _proc.is_alive()) for _proc in self.l_process])
    
    
    def close(
        self
    ) :
        
        if hasattr(self, "l_process") :
            
            pass
            
            #for proc in self.l_process:
            #    
            #    proc.join()
            #    proc.close()
    
    
    def createQueue(
        self,
    ) :
        
        self.close()
        
        self.queue = multiprocessing.Queue()
        
        #self.mp_manager = multiprocessing.Manager()
        #self.queue = self.mp_manager.Queue()
        
        self.l_process = []
        self.idleProcessIdx = 0
        
        for batchIdx in range(0, self.nBatch) :
            
            self.l_process.append(multiprocessing.Process(
                target = self.createBatch,
                args = (),
                kwargs = dict(
                    batchIdx = batchIdx,
                    procIdx = batchIdx,
                )
            ))
    
    
    def runQueue(
        self,
    ) :
        
        #print("Entering runQueue(...).")
        
        while(
            self.queue.qsize() < self.queueSize_max and
            self.countProcRunning() < self.parallelBatches_max and
            self.idleProcessIdx < len(self.l_process)
        ) :
            
            #print("Starting process idx %d." %(self.idleProcessIdx))
            self.l_process[self.idleProcessIdx].start()
            
            self.idleProcessIdx += 1
        
        #print("Exiting from runQueue(...).")
    
    
    def batchGenerator(
        self,
    ) :
        
        if (self.shuffleBatchesEveryEpoch) :
            
            self.rng.shuffle(self.a_sliceIdx_batched, axis = 0)
        
        self.createQueue()
        self.runQueue()
        print("In batchGenerator(...): Created and running batch queue...")
        
        #print("nProcess:", len(self.l_process))
        
        #while (self.queue.qsize() < min(10, max(1, int(self.queueSize_max/2)))) :
        #    
        #    True
        
        for iProc, proc in enumerate(self.l_process) :
            
            gc.collect()
            self.runQueue()
            
            #print("qsize:", self.queue.qsize())
            
            procIdx, result = self.queue.get()
            
            #print("Yielding batch...")
            #print(
            #    {_key: result[0][_key].shape for _key in result[0].keys()},
            #    result[1].shape,
            #)
            
            self.l_process[procIdx].join()
            self.l_process[procIdx].close()
            
            yield(result)
        
        
    
    
    def createBatch(
        self,
        batchIdx,
        procIdx,
        #queue,
    ) :
        
        d_data = sortedcontainers.SortedDict()
        d_data_sharedBuf = sortedcontainers.SortedDict()
        d_data_sharedBuf_name = sortedcontainers.SortedDict()
        d_branchName_alias = sortedcontainers.SortedDict()
        
        l_catNum = []
        l_sliceSize = []
        
        #for key in self.l_particleFeatureKey :
        for key in self.d_data_shape.keys() :
            
            l_temp = self.d_loadConfig[key]
            
            if type(l_temp) not in [list, tuple] :
                
                l_temp = [l_temp]
            
            #shape = (self.batchSize, self.d_loadConfig["maxParticles"], len(l_temp))
            #a_temp = numpy.empty(shape)
            a_temp = numpy.empty(self.d_data_shape[key])
            
            d_data_sharedBuf[key] = multiprocessing.shared_memory.SharedMemory(create = True, size = a_temp.nbytes)
            d_data[key] = numpy.ndarray(a_temp.shape, buffer = d_data_sharedBuf[key].buf)
            d_data_sharedBuf_name[key] = d_data_sharedBuf[key].name
            
            d_branchName_alias[key] = sortedcontainers.SortedDict({"%s%d" %(key, _i): _expr for _i, _expr in enumerate(l_temp)})
        
        #if (self.jetFeatureKey is not None) :
        #    
        #    key = self.jetFeatureKey
        #    
        #    l_temp = self.d_loadConfig[key]
        #    
        #    d_data_sharedBuf[key] = multiprocessing.shared_memory.SharedMemory(create = True, size = a_temp.nbytes)
        #    d_data[key] = numpy.ndarray(a_temp.shape, buffer = d_data_sharedBuf[key].buf)
        #    d_data_sharedBuf_name[key] = d_data_sharedBuf[key].name
        #    
        #    d_branchName_alias[key] = sortedcontainers.SortedDict({"%s%d" %(key, _i): _expr for _i, _expr in enumerate(l_temp)})
        
        #pprint.pprint(d_branchName_alias)
        
        #d_branchName_alias["points"]    = sortedcontainers.SortedDict({"coor%d" %(_i): _expr for _i, _expr in enumerate(self.d_loadConfig["points"])})
        #d_branchName_alias["features"]  = sortedcontainers.SortedDict({"feat%d" %(_i): _expr for _i, _expr in enumerate(self.d_loadConfig["features"])})
        #d_branchName_alias["mask"]      = sortedcontainers.SortedDict({"mask": self.d_loadConfig["mask"]})
        
        a_sliceIdx = self.a_sliceIdx_batched[batchIdx]
        
        nCpu = multiprocessing.cpu_count()
        nCpu_frac = float(self.d_loadConfig["cpuFrac"])
        #nCpu_use = int(nCpu_frac*nCpu)
        nCpu_use = int(nCpu_frac*nCpu/self.parallelBatches_max)
        #nCpu_use = min(int(nCpu_frac*nCpu), len(a_sliceIdx))
        
        nCpu_use = min(nCpu_use, len(a_sliceIdx))
        
        #print("nCpu_use:", nCpu_use)
        #print("# slice jobs:", len(a_sliceIdx))
        
        
        with multiprocessing.Pool(
            processes = nCpu_use,
            #maxtasksperchild = 1
        ) as pool :
            
            l_job = []
            
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
                    "category %d (%s), "
                    "sample %d, "
                    "file %d, "
                    %(
                    batchIdx, self.nBatch,
                    sliceIdx_batch, len(a_sliceIdx),
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
                ))
            
            
            pool.close()
            
            #print("[batch idx %d (num %d/%d)]" %(batchIdx, batchIdx+1, self.nBatch), "jobs:", l_job)
            
            l_isJobDone = [False] * len(l_job)
            
            while(False in l_isJobDone) :
                
                for iJob, job in enumerate(l_job) :
                    
                    if (job is None) :
                        
                        continue
                    
                    if (not l_isJobDone[iJob] and job.ready()) :
                        
                        l_isJobDone[iJob] = True
                        
                        retVal = job.get()
                        
                        #if (retVal) :
                        #    
                        #    print("retVal", retVal)
                        
                        l_job[iJob] = None
                        
                        if (not sum(l_isJobDone) % 10) :
                            
                            gc.collect()
                        
                        #print("[batch idx %d (num %d/%d)] processed output of read-job num %d (%d/%d done)." %(batchIdx, batchIdx+1, self.nBatch, iJob+1, sum(l_isJobDone), len(l_job)))
                        #print("=====> Memory:", utils.getMemoryMB())
            
            pool.join()
            pool.terminate()
            del pool
            
            for job in (l_job) :
                
                del job
            
            del l_job
        
        gc.collect()
        
        #print("Creating batch x-data.")
        d_data_cpy = sortedcontainers.SortedDict()
        
        #for key in self.l_particleFeatureKey :
        for key in self.d_data_shape.keys() :
            
            d_data_cpy[key] = d_data[key].copy()
            
            #s = numpy.sum(d_data_cpy[key])
            #
            #if (numpy.isnan(s) or numpy.isinf(s)) :
            #    
            #    print("-NAN-"*10)
            #    print(key)
            #    print(d_data_cpy[key].tolist())
            
            d_data_sharedBuf[key].close()
            d_data_sharedBuf[key].unlink()
            
        
        # One-hot labels
        #print("Creating batch y-data.")
        indices = numpy.repeat(l_catNum, l_sliceSize)
        a_label = numpy.zeros((len(indices), self.nCategory), dtype = numpy.int64)
        a_label[numpy.arange(len(indices)), indices] = 1
        
        #print("Adding result of batch idx %d (%d/%d) to queue." %(batchIdx, batchIdx+1, self.nBatch))
        
        self.queue.put(
            (procIdx, (d_data_cpy, a_label))
        )
