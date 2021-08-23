import argparse
import io
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
#import mplhep
import numpy
import os
import re
import sortedcontainers
import sparse
import tensorflow
import time
import uproot
import yaml

import ROOT


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


#@tensorflow.function
def sparse_to_dense(tensor) :
    
    #print(tensor)
    #print(type(tensor))
    #print(tensor.get_shape())
    
    #if (isinstance(tensor, tensorflow.sparse.SparseTensor) and tensor.get_shape()[0] is not None and tensor.get_shape()[1:] == (32, 32, 3)) :
    if (isinstance(tensor, tensorflow.sparse.SparseTensor)) :
        
        #print("Converting to dense")
        
        #print(tensor.get_shape())
        dn_tensor = tensorflow.sparse.to_dense(tensor)#, validate_indices = False)
        #print(dn_tensor)
        #print(type(dn_tensor))
        
        return dn_tensor
    
    elif (isinstance(tensor, sparse.COO)) :
        
        #print("Converting to dense...")
        
        print(tensor.shape)
        
        dn_tensor = tensorflow.constant(tensor.todense(), dtype = tensorflow.float16)
        
        print(dn_tensor)
        print(type(dn_tensor))
        
        return dn_tensor
    
    return tensor


cpalette_nipy_spectral = ColorPalette(
    a_r = numpy.array([0.0, 0.4667, 0.5333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7333, 0.9333, 1.0, 1.0, 1.0, 0.8667, 0.8, 0.8]),
    a_g = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4667, 0.6, 0.6667, 0.6667, 0.6, 0.7333, 0.8667, 1.0, 1.0, 0.9333, 0.8, 0.6, 0.0, 0.0, 0.0, 0.8]),
    a_b = numpy.array([0.0, 0.5333, 0.6, 0.6667, 0.8667, 0.8667, 0.8667, 0.6667, 0.5333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]),
    a_stop = numpy.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
)


uproot_lang = uproot.language.python.PythonLanguage()
uproot_lang.functions["min"] = numpy.minimum
uproot_lang.functions["max"] = numpy.maximum
uproot_lang.functions["where"] = numpy.where


#if (__name__ == "__main__") :
#    
#    exit()


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


def get_tfimage_roc(
    x,
    y,
    style,
    xlabel = "",
    ylabel = "",
) : 
    
    fig = matplotlib.pyplot.figure(figsize = [10, 10])
    axis = fig.add_subplot(1, 1, 1)
    
    #axis.set_aspect("equal", "box")
    
    axis.plot(x, y, style) 
    
    axis.set_xlim((0, 1))
    axis.set_ylim((1e-6, 1))
    
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    
    axis.set_yscale("log")
    axis.grid(True)
    
    fig.tight_layout()
    
    buf = io.BytesIO()
    matplotlib.pyplot.savefig(buf, format = "png")
    
    # Closing the figure prevents it from being displayed directly inside the notebook.
    matplotlib.pyplot.close(fig)
    
    buf.seek(0)
    
    # Convert PNG buffer to TF image
    tfimage = tensorflow.image.decode_png(buf.getvalue(), channels = 4)
    
    # Add the batch dimension
    tfimage = tensorflow.expand_dims(tfimage, 0)
    
    buf.close()
    
    return tfimage

