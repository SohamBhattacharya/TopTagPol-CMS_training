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
import particle_dataloader

#from tf_keras_model import get_particle_net, get_particle_net_lite#
import tf_keras_model

#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy("mixed_float16")
#mixed_precision.set_global_policy(policy)


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
    
    d_catInfo_trn = utils.load_catInfo_from_config(d_loadConfig)
    d_catInfo_tst = utils.load_catInfo_from_config(d_loadConfig)
    
    
    # dict.copy() will not copy the item lists
    d_sliceInfo_trn = copy.deepcopy(utils.d_sliceInfo_template)
    d_sliceInfo_tst = copy.deepcopy(utils.d_sliceInfo_template)
    
    nCpu = multiprocessing.cpu_count()
    nCpu_frac = float(d_loadConfig["cpuFrac"])
    nCpu_use = int(nCpu_frac*nCpu)
    
    pool = multiprocessing.Pool(processes = nCpu_use, maxtasksperchild = 1)
    l_job = []
    
    for (iCat, cat) in enumerate(d_catInfo_trn.keys()) :
        
        catInfo_trn = d_catInfo_trn[cat]
        catInfo_tst = d_catInfo_tst[cat]
        
        for iSample, l_fileAndTreeName in enumerate(catInfo_trn.l_sample_fileAndTreeName):
            
            l_job.append(pool.apply_async(
                particle_dataloader.loadSliceInfo,
                (),
                dict(
                    catNum = cat,
                    sampleIdx = iSample,
                    catInfo = catInfo_trn,
                    sliceSize = sliceSize,
                ),
            ))
    
    pool.close()
    
    l_isJobDone = [False] * len(l_job)
    
    while(False in l_isJobDone) :
        
        for iJob, job in enumerate(l_job) :
            
            if (job is None) :
                
                continue
            
            if (not l_isJobDone[iJob] and job.ready()) :
                
                l_isJobDone[iJob] = True
                
                d_result = job.get()
                
                catNum              = d_result["catNum"]
                sampleIdx           = d_result["sampleIdx"]
                nJet_sample         = d_result["nJet_sample"]
                nSlice_sample       = d_result["nSlice_sample"]
                d_sliceInfo_sample  = d_result["d_sliceInfo_sample"]
                
                catInfo_trn = d_catInfo_trn[catNum]
                catInfo_tst = d_catInfo_tst[catNum]
                
                nSlice_tst_sample = int(d_loadConfig["testFraction"] * nSlice_sample)
                nSlice_trn_sample = nSlice_sample - nSlice_tst_sample
                
                for key in d_sliceInfo_trn.keys() :
                    
                    #print("$"*50, key, len(d_sliceInfo_sample[key][0: nSlice_trn_sample]))
                    #print("&"*50, key, len(d_sliceInfo_sample[key][nSlice_trn_sample:]))
                    d_sliceInfo_trn[key].extend(d_sliceInfo_sample[key][0: nSlice_trn_sample])
                    d_sliceInfo_tst[key].extend(d_sliceInfo_sample[key][nSlice_trn_sample:])
                
                catInfo_trn.l_sample_nJet[sampleIdx] += nSlice_trn_sample*sliceSize
                catInfo_tst.l_sample_nJet[sampleIdx] += nSlice_tst_sample*sliceSize
                
                l_job[iJob] = None
                
                print("Loaded slices for [cat %d, sample %d]. Jobs done: %d/%d." %(catNum, sampleIdx, sum(l_isJobDone), len(l_isJobDone)))
                
                if (not sum(l_isJobDone) % 10) :
                    
                    gc.collect()
    
    pool.join()
    
    
    print(catInfo_trn.l_sample_nJet)
    print(catInfo_tst.l_sample_nJet)
    
    
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
    
    dataloader_trn = particle_dataloader.ParticleDataset(
        d_loadConfig = d_loadConfig,
        d_catInfo = d_catInfo_trn,
        d_sliceInfo = d_sliceInfo_trn,
        a_sliceIdx_batched = a_sliceIdx_shf_batched,
        l_featureKey = l_featureKey,
        sliceSize = sliceSize,
        batchSize = batchSize,
        #nBatch = nBatch,
        shuffleBatchesEveryEpoch = True,
    )
    
    
    
    
    dataloader_tst = particle_dataloader.ParticleDataset(
        d_loadConfig = d_loadConfig,
        d_catInfo = d_catInfo_trn,
        d_sliceInfo = d_sliceInfo_tst,
        a_sliceIdx_batched = [numpy.arange(0, nSlice_tst)],
        l_featureKey = l_featureKey,
        sliceSize = sliceSize,
        batchSize = nSlice_tst*sliceSize,
        #nBatch = nBatch,
        shuffleBatchesEveryEpoch = False,
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
    )#.prefetch(tensorflow.data.AUTOTUNE)
    
    
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
        #profile_batch = int(nBatch/2),
        profile_batch = (nBatch//5, 2*(nBatch//5)),
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
