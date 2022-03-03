from __future__ import print_function

#import mxnet
import argparse
import awkward
import collections
import copy
#import cppyy
#import cppyy.ll
import comet_ml
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

import callbacks
#import networks
import utils

import ParticleDataLoader

#from tf_keras_model import get_particle_net, get_particle_net_lite#
import tf_keras_model


@dataclasses.dataclass
class CategoryInfo :
    
    catNum:         int
    catName:        str
    l_sample:       List[str]
    cut:            str
    
    def __post_init__(
        self,
    ) :
        
        self.nJet = 0
        
        self.l_sample_fileAndTreeName = [None] * len(self.l_sample)
        self.l_sample_weight = [None] * len(self.l_sample)
        self.l_sample_nJet = [0] * len(self.l_sample)
        self.l_sample_nJetMax = [0] * len(self.l_sample)
        
        for iSample, sample in enumerate(self.l_sample) :
            
            l_token = sample.strip().split(":")
            
            fName = l_token[0]
            treeName = l_token[1]
            weight = float(l_token[2])
            nJetMax = int(l_token[3])
            
            #self.l_sample_fileAndTreeName[iSample] = ["%s:%s" %(ele, treeName) for ele in numpy.loadtxt(fName, dtype = str, delimiter = "*"*100)]
            self.l_sample_fileAndTreeName[iSample] = ["root://dcache-cms-xrootd.desy.de/%s:%s" %(ele, treeName) for ele in numpy.loadtxt(fName, dtype = str, delimiter = "*"*100)]
            self.l_sample_weight[iSample] = weight
            self.l_sample_nJetMax[iSample] = nJetMax


def get_sample_table_str(d_catInfo) :
    
    l_table = []
    
    l_table.append([
        "Category",
        "Sample",
        "Jet count",
        "Jet fraction",
    ])
    
    nJetTotal = sum([sum(catInfo.l_sample_nJet) for catInfo in d_catInfo.values()])
    
    for (iCat, cat) in enumerate(d_catInfo.keys()) :
        
        catInfo = d_catInfo[cat]
        
        rowLabel = "%d (%s)" %(catInfo.catNum, catInfo.catName)
        
        for iSample, sample in enumerate(catInfo.l_sample) :
            
            l_table.append([
                rowLabel if (not iSample) else "",
                sample,
                catInfo.l_sample_nJet[iSample],
                "%0.4f" %(catInfo.l_sample_nJet[iSample]/nJetTotal),
            ])
        
        nJet = sum(catInfo.l_sample_nJet)
        
        l_table.append([
            "**%s**" %(rowLabel),
            "**Total**",
            "**%d**" %(nJet),
            "**%0.4f**" %(nJet/nJetTotal),
        ])
    
    l_table.append([
        "**All**",
        "--",
        "**%d**" %(nJetTotal),
        "--",
    ])
    
    
    table_str = tabulate.tabulate(
        l_table,
        headers = "firstrow",
        #tablefmt = "fancy_grid"
        tablefmt = "github"
    )
    
    #print(table_str)
    
    return table_str


def main() :
    
    #cometml_experiment = comet_ml.Experiment()
    
    
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
    
    network_fn_name = "get_%s" %(d_loadConfig["network"])
    assert(hasattr(tf_keras_model, network_fn_name))
    
    nSlicePerBatch = int(batchSize/sliceSize)
    nCategory = len(d_loadConfig["categories"])
    
    l_featureKey = ["points", "features", "mask"]
    
    for key in l_featureKey :
        
        assert(key in d_loadConfig)
    
    d_catInfo_trn = sortedcontainers.SortedDict()
    d_catInfo_tst = sortedcontainers.SortedDict()
    
    for iCat, cat in enumerate(d_loadConfig["categories"]) :
        
        d_catInfo_trn[iCat] = CategoryInfo(
            catNum = iCat,
            catName = cat["name"],
            l_sample = cat["samples"],
            cut = cat["cut"],
        )
        
        d_catInfo_tst[iCat] = CategoryInfo(
            catNum = iCat,
            catName = cat["name"],
            l_sample = cat["samples"],
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
    
    # dict.copy() will not copy the item lists
    d_sliceInfo_trn = copy.deepcopy(d_sliceInfo)
    d_sliceInfo_tst = copy.deepcopy(d_sliceInfo)
    
    nCpu = multiprocessing.cpu_count()
    nCpu_frac = float(d_loadConfig["cpuFrac"])
    nCpu_use = int(nCpu_frac*nCpu)
    
    for (iCat, cat) in enumerate(d_catInfo_trn.keys()) :
        
        catInfo_trn = d_catInfo_trn[cat]
        catInfo_tst = d_catInfo_tst[cat]
        
        for iSample, sample_fileAndTreeName in enumerate(catInfo_trn.l_sample_fileAndTreeName):
            
            nJet_sample = 0
            nSlice_sample = 0
            
            d_sliceInfo_sample = copy.deepcopy(d_sliceInfo)
            
            #for (iTree, fileAndTreeName) in enumerate(sample_fileAndTreeName[0: 2]) :
            for (iTree, fileAndTreeName) in enumerate(sample_fileAndTreeName) :
                
                print("Counting jets in: %s" %(fileAndTreeName))
                
                with uproot.open(
                    fileAndTreeName,
                    #num_workers = nCpu_use,
                    #num_fallback_workers = nCpu_use,
                    xrootd_handler = uproot.MultithreadedXRootDSource,
                    timeout = None,
                ) as tree :
                    
                    brName = "cut"
                    
                    branches = tree.arrays(
                        expressions = [brName],
                        cut = catInfo_trn.cut,
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
                    
                    if (catInfo_trn.l_sample_nJetMax[iSample] > 0 and nJet_sample >= catInfo_trn.l_sample_nJetMax[iSample]) :
                        
                        break
                    
                    #print("counts", l)
            
            
            nSlice_tst_sample = int(d_loadConfig["testFraction"] * nSlice_sample)
            nSlice_trn_sample = nSlice_sample - nSlice_tst_sample
            #print("*"*50, iCat, iSample, nSlice_trn_sample, nSlice_tst_sample, nSlice_sample, len(d_sliceInfo_sample["a_sampleIdx"]))
            
            for key in d_sliceInfo_trn.keys() :
                
                #print("$"*50, key, len(d_sliceInfo_sample[key][0: nSlice_trn_sample]))
                #print("&"*50, key, len(d_sliceInfo_sample[key][nSlice_trn_sample:]))
                d_sliceInfo_trn[key].extend(d_sliceInfo_sample[key][0: nSlice_trn_sample])
                d_sliceInfo_tst[key].extend(d_sliceInfo_sample[key][nSlice_trn_sample:])
            
            catInfo_trn.l_sample_nJet[iSample] += nSlice_trn_sample*sliceSize
            catInfo_tst.l_sample_nJet[iSample] += nSlice_tst_sample*sliceSize
            
            #print("#"*50, iCat, iSample, len(d_sliceInfo_trn["a_sampleIdx"]), len(d_sliceInfo_tst["a_sampleIdx"]))
    
    
    
    nSlice_trn = len(d_sliceInfo_trn["a_sampleIdx"])
    nSlice_tst = len(d_sliceInfo_tst["a_sampleIdx"])
    
    nSlice_trn_orig = nSlice_trn
    
    # Choose nSlice slices such that nSlice*sliceSize is a multiple of batchSize
    nBatch = int(nSlice_trn/nSlicePerBatch)
    nSlice_trn = nSlicePerBatch * nBatch
    nSlice_dropped_trn = nSlice_trn_orig - nSlice_trn
    
    # Adjust the jet count for the new value of nSlice
    for iSlice in range(nSlice_trn, nSlice_trn_orig) :
        
        catNum = d_sliceInfo_trn["a_catNum"][iSlice]
        sampleIdx = d_sliceInfo_trn["a_sampleIdx"][iSlice]
        
        d_catInfo_trn[catNum].l_sample_nJet[sampleIdx] -= sliceSize
    
    #nBatch = int(numpy.ceil(nSlice_trn/nSlicePerBatch))
    
    print("nSlice_trn:", nSlice_trn)
    print("nSlice_tst:", nSlice_tst)
    print("sliceSize:", sliceSize)
    print("nBatch:", nBatch)
    print("batchSize:", batchSize)
    print("nSlicePerBatch:", nSlicePerBatch)
    print("Dropped training slices:", nSlice_dropped_trn)
    
    
    # Shuffle slices
    a_sliceIdx_shf = numpy.arange(0, nSlice_trn)
    numpy.random.shuffle(a_sliceIdx_shf)
    
    #a_sliceIdx_shf_batched = numpy.split(a_sliceIdx_shf, nBatch)
    a_sliceIdx_shf_batched = numpy.split(a_sliceIdx_shf, range(nSlicePerBatch, nSlice_trn, nSlicePerBatch))
    
    dataloader_trn = ParticleDataLoader.ParticleDataset(
        d_loadConfig = d_loadConfig,
        d_catInfo = d_catInfo_trn,
        d_sliceInfo = d_sliceInfo_trn,
        a_sliceIdx_batched = a_sliceIdx_shf_batched,
        l_featureKey = l_featureKey,
        sliceSize = sliceSize,
        batchSize = batchSize,
        #nBatch = nBatch,
    )
    
    
    
    
    dataloader_tst = ParticleDataLoader.ParticleDataset(
        d_loadConfig = d_loadConfig,
        d_catInfo = d_catInfo_trn,
        d_sliceInfo = d_sliceInfo_tst,
        a_sliceIdx_batched = [numpy.arange(0, nSlice_tst)],
        l_featureKey = l_featureKey,
        sliceSize = sliceSize,
        batchSize = nSlice_tst*sliceSize,
        #nBatch = nBatch,
    )
    
    
    print("Loading testing data...")
    dataset_tst = next(dataloader_tst.batchGenerator())
    dataloader_tst.close()
    print("Loaded testing data.")
    #print(dataset_tst[0])
    #print(dataset_tst[1])
    #print(dataset_tst[1].shape)
    
    
    dataset_trn = tensorflow.data.Dataset.from_generator(
        dataloader_trn.batchGenerator,
        output_types = (
            {_key: float for _key in dataloader_trn.d_data_outputShape},
            tensorflow.int64
        ),
        output_shapes = (
            dataloader_trn.d_data_outputShape,
            (None, nCategory)
        ),
    ).prefetch(tensorflow.data.AUTOTUNE)
    
    
    # Create tensorflow file writers
    tensorboard_file_writer_trn = tensorflow.summary.create_file_writer("%s/train" %(tensorboard_dir))
    tensorboard_file_writer_tst = tensorflow.summary.create_file_writer("%s/validation" %(tensorboard_dir))
    tensorboard_file_writer_params = tensorflow.summary.create_file_writer("%s/params" %(tensorboard_dir))
    tensorboard_file_writer_config = tensorflow.summary.create_file_writer("%s/config" %(tensorboard_dir))
    
    
    # Log sample data
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
    
    
    #model = get_particle_net(nCategory, input_shapes = dataloader_trn.d_data_inputShape)
    #model = get_particle_net_lite(nCategory, input_shapes = dataloader_trn.d_data_inputShape)
    model = getattr(tf_keras_model, network_fn_name)(nCategory, input_shapes = dataloader_trn.d_data_inputShape)
    
    lr_boundaries = [_lrEpoch * nBatch for _lrEpoch in d_loadConfig["learningRate"]["epoch"]]
    lr_values = d_loadConfig["learningRate"]["rate"]
    
    lr_schedule = tensorflow.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = lr_boundaries,
        values = lr_values,
    )
    
    optimizer = tensorflow.keras.optimizers.Adam(
        learning_rate = lr_schedule
    )
    
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = optimizer,
        metrics = [
            "accuracy",
        ],
    )
    
    model_summary_str = []
    model.summary(print_fn = lambda x: model_summary_str.append(x))
    model_summary_str = "\n".join(model_summary_str)
    print(model_summary_str)
    
    
    # Log the input configuration and model summary
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
    
    
    checkpoint_file = "%s/weights_epoch-{epoch:d}.hdf5" %(checkpoint_dir)
    
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
    
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
        log_dir = tensorboard_dir,
        histogram_freq = 1,
        write_graph = True,
        write_images = False,
        write_steps_per_second = False,
        #update_freq = "epoch",
        update_freq = "batch",
        #update_freq = 10,
        #profile_batch = (1, nBatch_trn),
        #profile_batch = nBatch_trn,
        profile_batch = int(nBatch/2),
        embeddings_freq = 0,
        embeddings_metadata = None,
    )
    
    mylog_callback = callbacks.CustomCallback(
        model = model,
        writer = tensorboard_file_writer_tst,
        writerPath = "%s/validation" %(tensorboard_dir),
        dataset = dataset_tst,
        l_categoryLabel = (d_catInfo_trn.keys()),
    )
    
    
    model.fit(
        x = dataset_trn,
        epochs = d_loadConfig["nEpoch"],
        validation_data = dataset_tst,
        shuffle = False,
        callbacks = [checkpoint_callback, tensorboard_callback, mylog_callback],
    )
    
    
    return 0


if (__name__ == "__main__") :
    
    main()
