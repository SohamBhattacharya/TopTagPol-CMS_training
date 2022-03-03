import argparse
import copy
import dataclasses
import io
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
#import mplhep
import numpy
import os
import psutil
import re
import sortedcontainers
import sparse
import subprocess
#import tensorflow
import time
import uproot
import yaml

import ROOT

from typing import List, Set, Dict, Tuple, Optional


uproot_lang = uproot.language.python.PythonLanguage()
uproot_lang.functions["min"] = numpy.minimum
uproot_lang.functions["max"] = numpy.maximum
uproot_lang.functions["where"] = numpy.where
uproot_lang.functions["sum"] = numpy.sum


d_sliceInfo_template = {
    "a_catNum":             [],
    
    "a_sampleIdx":          [],
    "a_fileIdx":            [],
    
    "a_evtIdx_start":       [],
    "a_jetIdx_start":       [],
    
    "a_evtIdx_end":         [],
    "a_jetIdx_end":         [],
}


class ColorPalette :
    
    def __init__(
        self,
        a_r,
        a_g,
        a_b,
        a_stop,
    ) :
        
        
        self.a_r = a_r
        self.a_g = a_g
        self.a_b = a_b
        self.a_stop = a_stop
        
        self.nStop = len(a_stop)
    
    def set(self, nContour = 500) :
        
        ROOT.gStyle.SetNumberContours(nContour)
        ROOT.TColor.CreateGradientColorTable(self.nStop, self.a_stop, self.a_r, self.a_g, self.a_b, nContour)


cpalette_nipy_spectral = ColorPalette(
    a_r = numpy.array([0.0, 0.4667, 0.5333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7333, 0.9333, 1.0, 1.0, 1.0, 0.8667, 0.8, 0.8]),
    a_g = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4667, 0.6, 0.6667, 0.6667, 0.6, 0.7333, 0.8667, 1.0, 1.0, 0.9333, 0.8, 0.6, 0.0, 0.0, 0.0, 0.8]),
    a_b = numpy.array([0.0, 0.5333, 0.6, 0.6667, 0.8667, 0.8667, 0.8667, 0.6667, 0.5333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]),
    a_stop = numpy.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
)


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
        self.l_sample_weight = [1] * len(self.l_sample)
        self.l_sample_nJet = [0] * len(self.l_sample)
        self.l_sample_nJetMax = [-1] * len(self.l_sample)
        
        for iSample, sample in enumerate(self.l_sample) :
            
            l_token = sample.strip().split(":")
            
            fName = l_token[0]
            treeName = l_token[1]
            
            if (".txt" in fName) :
                
                #self.l_sample_fileAndTreeName[iSample] = ["%s:%s" %(ele, treeName) for ele in numpy.loadtxt(fName, dtype = str, delimiter = "*"*100)]
                self.l_sample_fileAndTreeName[iSample] = ["root://dcache-cms-xrootd.desy.de/%s:%s" %(_ele, treeName) for _ele in numpy.loadtxt(fName, dtype = str, delimiter = "*"*100)]
            
            elif (".root" in fName) :
                
                self.l_sample_fileAndTreeName[iSample] = ["root://dcache-cms-xrootd.desy.de/%s:%s" %(fName, treeName)]
            
            else :
                
                print("Invalid entry for sample: %s" %(sample))
                exit(1)
            
            if (len(l_token) > 2) :
                
                weight = float(l_token[2])
                self.l_sample_weight[iSample] = weight
            
            if (len(l_token) > 3) :
                
                nJetMax = int(l_token[3])
                self.l_sample_nJetMax[iSample] = nJetMax


def load_catInfo_from_config(
    d_loadConfig
) :
    
    d_catInfo = sortedcontainers.SortedDict()
    
    for iCat, cat in enumerate(d_loadConfig["categories"]) :
        
        d_catInfo[iCat] = CategoryInfo(
            catNum = iCat,
            catName = cat["name"],
            l_sample = cat["samples"],
            cut = cat["cut"],
        )
    
    return d_catInfo


def replace_in_list(l, find, repl) :
    
    return [s.replace(find, repl) for s in l]
    
    for idx in range(0, len(l)) :
        
        if (isinstance(d[key], str)) :
            
            d[key] = d[key].replace(find, repl)
        
        elif (isinstance(d[key], list)) :
            
            d[key] = replace_in_list(d[key], find, repl)
        
        elif (isinstance(d[key], dict)) :
            
            replace_in_dict(d[key], find, repl)


def replace_in_dict(d, find, repl) :
    
    for key in d :
        
        if (isinstance(d[key], str)) :
            
            d[key] = d[key].replace(find, repl)
        
        elif (isinstance(d[key], list)) :
            
            d[key] = replace_in_list(d[key], find, repl)
        
        elif (isinstance(d[key], dict)) :
            
            replace_in_dict(d[key], find, repl)
    
    
    #return d


def load_config(fileName) :
    
    with open(fileName, "r") as fopen :
        
        fileContent = fopen.read()
        print("Loading config:")
        print(fileContent)
        
        d_loadConfig = yaml.load(fileContent, Loader = yaml.FullLoader)
        
        if ("jetName" in d_loadConfig.keys()) :
            
            jetNameKey = d_loadConfig["jetName"].split(":")[0]
            jetName = d_loadConfig["jetName"].split(":")[1]
            
            fileContent = fileContent.replace(jetNameKey, jetName)
        
        d_loadConfig = yaml.load(fileContent, Loader = yaml.FullLoader)
        
        d_loadConfig["fileContent"] = fileContent
        
        return d_loadConfig


def run_cmd_list(l_cmd) :
    
    for cmd in l_cmd :
        
        retval = os.system(cmd)
        
        if (retval) :
            
            exit()


def getMemoryMB(process = -1) :
    
    if (process < 0) :
        
        process = psutil.Process(os.getpid())
    
    mem = process.memory_info().rss / 1024.0**2
    
    return mem


def get_fileAndTreeNames(in_list) :
    
    fileAndTreeNames = []
    
    for fName in in_list :
        
        if (".root" in fName) :
            
            fileAndTreeNames.append(fName)
        
        elif (".txt" in fName) :
            
            sourceFile, treeName = fName.strip().split(":")
            
            rootFileNames = numpy.loadtxt(sourceFile, dtype = str, delimiter = "*"*100)
            
            for rootFileName in rootFileNames :
                
                fileAndTreeNames.append("%s:%s" %(rootFileName, treeName))
        
        else :
            
            print("Error. Invalid syntax for fileAndTreeNames: %s" %(fName))
            exit(1)
    
    return fileAndTreeNames


def format_file(filename, d, execute = False) :
    
    l_cmd = []
    
    for key in d :
        
        val = d[key]
        
        l_cmd.append("sed -i \"s#{find}#{repl}#g\" {filename}".format(
            find = key,
            repl = val,
            filename = filename,
        ))
    
    if (execute) :
        
        run_cmd_list(l_cmd)
    
    else :
        
        return l_cmd


def get_name_withtimestamp(dirname) :
    
    if (os.path.exists(dirname)) :
        
        timestamp = subprocess.check_output(["date", "+%Y-%m-%d_%H-%M-%S", "-r", dirname]).strip()
        timestamp = timestamp.decode("UTF-8") # Convert from bytes to string
        
        dirname_new = "%s_%s" %(dirname, str(timestamp))
        
        return dirname_new
    
    return None


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
