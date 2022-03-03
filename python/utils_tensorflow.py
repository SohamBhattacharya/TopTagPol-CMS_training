import argparse
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
import tensorflow
import time
import uproot
import yaml

import ROOT


@tensorflow.function
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



def get_tfimage_xyplot(
    l_plotdata,
    xlim = None,
    ylim = None,
    logx = False,
    logy = False,
    xlabel = "",
    ylabel = "",
) : 
    
    # Each element of l_data must be of the form: [x, y, fmt, plot_options_dict]
    # plot_options_dict is a dictionary with valid matplotlip plot options: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    # Example: [[1, 2], [1, 2], "ro", {"style": "-", "color": "r", "label": "curve1"}]
    
    fig = matplotlib.pyplot.figure(figsize = [5, 5])
    axis = fig.add_subplot(1, 1, 1)
    
    #axis.set_aspect("equal", "box")
    
    for plotdata in l_plotdata :
        
        axis.plot(plotdata[0], plotdata[1], plotdata[2], **plotdata[3]) 
    
    if (xlim is not None) :
        
        axis.set_xlim(xlim)
    
    if (ylim is not None) :
        
        axis.set_ylim(ylim)
    
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    
    if (logx) :
        
        axis.set_xscale("log")
    
    if (logy) :
        
        axis.set_yscale("log")
    
    axis.grid(True)
    axis.legend()
    
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
