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
import particle_dataloader


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
        required = False,
        default = None,
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
    
    #print(d_model)
    
    fileAndTreeNames = utils.get_fileAndTreeNames(args.inFileNames) if (args.inFileNames) is not None else d_config["samples"]
    
    nFile = len(fileAndTreeNames)
    
    catInfo = utils.CategoryInfo(
        catNum = 0, # Just a dummy value, has no impact for evaluation
        catName = "category", # Just a dummy value, has no impact for evaluation
        l_sample = fileAndTreeNames,
        cut = None,
    )
    
    if ("/" in args.outFileName) :
        
        outDir = args.outFileName[0: args.outFileName.rfind("/")]
        os.system("mkdir -p %s" %(outDir))
    
    outFileName = "file:%s" %(args.outFileName) if (args.outFileName.find("file:") != 0) else args.outFileName
    outFile = ROOT.TFile.Open(outFileName, "RECREATE")
    outTree = ROOT.TTree("tree", "tree")
    
    d_classifierBranch = {}
    
    for modelKey in d_model.keys() :
        
        for classifierKey in d_model[modelKey]["classifiers"].keys() :
            
            br_title = (
                "\nmodelName: {modelName}; "
                "\nmodelConfig: {modelConfig}; "
            ).format(
                **d_config
            )
            
            d_classifierBranch[classifierKey] = ROOT.std.vector("double")()
            br = outTree.Branch(classifierKey, d_classifierBranch[classifierKey])
            br.SetTitle(br_title)
    
    
    #print(catInfo.__dict__)
    print(d_classifierBranch)
    
    l_featureKey = ["points", "features", "mask"]
    
    d_branchName_alias = sortedcontainers.SortedDict()
    d_branchName_alias_flat = sortedcontainers.SortedDict()
    
    for ftKey in l_featureKey :
        
        l_temp = d_modelConfig[ftKey]
        
        if type(l_temp) not in [list, tuple] :
            
            l_temp = [l_temp]
        
        d_branchName_alias[ftKey] = sortedcontainers.SortedDict({"%s%d" %(ftKey, _i): _expr for _i, _expr in enumerate(l_temp)})
        #print(ftKey, d_branchName_alias[ftKey])
        d_branchName_alias_flat.update(d_branchName_alias[ftKey])
    
    
    #print(d_branchName_alias_flat)
    
    for iSample, l_fileAndTreeName in enumerate(catInfo.l_sample_fileAndTreeName) :
        
        nFile = len(l_fileAndTreeName)
        
        for iFile, fileAndTreeName in enumerate(l_fileAndTreeName) :
            
            verbosetag = "[sample %d/%d, file %d/%d]" %(iSample+1, len(catInfo.l_sample_fileAndTreeName), iFile+1, nFile)
            
            print("%s input : %s " %(verbosetag, fileAndTreeName))
            print("%s output: %s " %(verbosetag, outFileName))
            #exit()
            
            with uproot.open(
                fileAndTreeName,
                xrootd_handler = uproot.MultithreadedXRootDSource,
                timeout = None,
            ) as tree :
                
                nEvent_file = tree.num_entries
            
            nEvent_processed = 0
            
            d_data = sortedcontainers.SortedDict()
            
            for branches in uproot.iterate(
                files = fileAndTreeName,
                expressions = list(d_branchName_alias_flat.keys()),
                cut = None,
                #entry_start = evtIdx_start,
                #entry_stop = evtIdx_end+1,
                aliases = d_branchName_alias_flat,
                language = utils.uproot_lang,
                step_size = 1000,
                num_workers = 20,
                xrootd_handler = uproot.MultithreadedXRootDSource,
            ) :
                
                ##if (nEvent_processed >= 100) : break
                
                bookKeepKey = list(d_branchName_alias_flat.keys())[0]
                bookKeepBranch = branches[bookKeepKey]
                
                nEvent = len(bookKeepBranch)
                
                for iEvent in range(0, nEvent):
                    
                    ##if (nEvent_processed >= 100) : break
                    
                    # Clear branches
                    for classifierKey in d_classifierBranch.keys() :
                        
                        d_classifierBranch[classifierKey].clear()
                    
                    nJet = len(bookKeepBranch[iEvent])
                    
                    if (nJet) :
                        
                        d_data = sortedcontainers.SortedDict()
                        
                        for ftKey in l_featureKey :
                            
                            shape = (nJet, d_modelConfig["maxParticles"], len(d_branchName_alias[ftKey]))
                            d_data[ftKey] = numpy.empty(shape)
                            
                            #print(branches)
                            
                            for iBrKey, brKey in enumerate(list(d_branchName_alias[ftKey].keys())) :
                                
                                #print(brKey)
                                
                                #a_br = awkward.flatten(branches[brKey][iEvent], axis = 1)
                                a_br = branches[brKey][iEvent]
                                
                                #print(nEvent_processed, nJet, brKey, len(a_br), a_br)
                                
                                a_br_padded = awkward.pad_none(a_br, target = d_modelConfig["maxParticles"], axis = 1, clip = True)
                                a_br_masked = awkward.fill_none(a_br_padded, 0)
                                
                                d_data[ftKey][:, :, iBrKey] = a_br_masked
                        
                        d_ypred = {}
                        
                        for modelKey in d_model.keys() :
                            
                            y_pred = d_model[modelKey]["model"].predict(d_data)
                            
                            if (not len(d_ypred)) :
                                
                                nNode = y_pred.shape[1]
                                
                                for iNode in range(0, nNode) :
                                    
                                    d_ypred["node%d" %(iNode)] = "y_pred[:, %d]" %(iNode)
                            
                            for classifierKey in d_model[modelKey]["classifiers"].keys() :
                                
                                eval_str = d_model[modelKey]["classifiers"][classifierKey].format(**d_ypred)
                                arr_classifier = eval(eval_str).astype(float)
                                
                                # Fill branches
                                for val in arr_classifier :
                                    
                                    d_classifierBranch[classifierKey].push_back(val)
                                
                                #print(modelKey, classifierKey, arr_classifier, nJet)
                            
                    
                    
                    # Fill the tree for each event
                    outTree.Fill()
                    nEvent_processed += 1
                    
                    if (nEvent_processed == 1 or not (nEvent_processed % d_config["printEvery"]) or nEvent_processed == nEvent_file) :
                        
                        print("%s: processed event %d/%d." %(verbosetag, nEvent_processed, nEvent_file))
    
    
    print("Processing successful. Saving output tree...")
    
    outFile.cd()
    outTree.Write()
    outFile.Close()
    
    print("Output tree successfully saved.")
    
    return 0



if (__name__ == "__main__") :
    
    main()
