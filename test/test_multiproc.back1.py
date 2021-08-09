from __future__ import print_function

#import mxnet
import collections
import concurrent.futures
import gc
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import memory_profiler
import multiprocessing
import multiprocessing.managers
import numpy
import os
import PIL
import psutil
import sortedcontainers
import sys
import tensorflow
import time
import uproot

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


class MyManager(multiprocessing.managers.BaseManager) :
    
    pass


MyManager.register("SortedDict", sortedcontainers.SortedDict)


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
        
        self.xBinSize = (self.xMax-self.xMin) / nBinX
        self.yBinSize = (self.yMax-self.yMin) / nBinY


class CategoryInfo :
    
    def __init__(
        self,
        catNum,
        l_fileAndTreeName,
        l_brName,
        cut
    ) :
        
        self.catNum = catNum
        self.l_fileAndTreeName = l_fileAndTreeName
        self.l_brName = l_brName
        self.cut = cut
        
        #self.l_fileAndTreeName = l_fileAndTreeName
        
        #self.l_tree.append(uproot.open("%s:%s" %(fileName, treeName), num_workers = 10))
        #self.tree = uproot.lazy("%s:%s" %(fileName, treeName))
        
        #self.branches = self.tree.arrays(
        #    expressions = l_brName,
        #    cut = cut,
        #)
        
        #self.branches = uproot.lazy(
        #    files = [self.tree]
        #    expressions = l_brName,
        #    cut = cut,
        #)
        
        self.l_tree = []
        
        for name in l_fileAndTreeName :
            
            self.l_tree.append(uproot.open(
                path = name,
            ))
        
        
        self.nObj = 0


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


def sparse_to_dense(tensor) :
    
    print(tensor)
    print(type(tensor))
    print(tensor.get_shape())
    
    #if (isinstance(tensor, tensorflow.sparse.SparseTensor) and tensor.get_shape()[0] is not None and tensor.get_shape()[1:] == (32, 32, 3)) :
    if (isinstance(tensor, tensorflow.sparse.SparseTensor)) :
        
        print("Converting to dense")
        dn_tensor = tensorflow.sparse.to_dense(tensor)#, validate_indices = False)
        print(dn_tensor)
        print(type(dn_tensor))
        
        return dn_tensor
    
    return tensor


def computeBins(
    tree,
    l_brName,
    cut,
    entry_start,
    entry_stop,
    #d_arrayIdx,
    #l_label,
    offset,
    imgSpec,
    debugId = None,
    printEvery = 1000,
) :
    
    if (entry_start >=  entry_stop) :
        
        return 0
    
    branches = tree.arrays(
        expressions = l_brName,
        cut = cut,
        entry_start = entry_start,
        entry_stop = entry_stop,
    )
    
    nEvent = len(branches[l_brName[0]])
    #nEvent = 1000
    
    jetIdx = -1
    nJet_sel = 0
    
    
    nBinX = imgSpec.nBinX
    nBinY = imgSpec.nBinY
    
    xMin = imgSpec.xMin
    xMax = imgSpec.xMax
    
    yMin = imgSpec.yMin
    yMax = imgSpec.yMax
    
    xBinSize = imgSpec.xBinSize
    yBinSize = imgSpec.yBinSize
    
    
    d_arrayIdx = sortedcontainers.SortedDict()
    #d_arrayIdx = {}
    
    for iEvent in range(0, nEvent) :
        
        nJet = len(branches[l_brName[0]][iEvent])
        
        for iJet in range(0, nJet) :
            
            jetIdx += 1
            #l_label.append(catNum)
            
            #.astype(dtype = numpy.float16, copy = False)
            
            a_x  = ((branches["fatJet_constiTrans_x_reco"][iEvent][iJet].to_numpy() - xMin) / xBinSize).astype(dtype = numpy.uint32)
            a_y  = ((branches["fatJet_constiTrans_y_reco"][iEvent][iJet].to_numpy() - yMin) / yBinSize).astype(dtype = numpy.uint32)
            a_w  = branches["fatJet_constiTrans_w_reco"][iEvent][iJet].to_numpy().astype(dtype = numpy.float16, copy = False)
            
            #a_id = vec_pdgid2layer(numpy.int_(branches["fatJet_constiTrans_id_reco"][iEvent][iJet].to_numpy()))
            a_id = vec_pdgid2layer(branches["fatJet_constiTrans_id_reco"][iEvent][iJet].to_numpy())
            
            nConsti = len(a_x)
            
            a_jetIdx = numpy.full(nConsti, fill_value = offset+jetIdx, dtype = numpy.uint32)
            
            
            a_constiIdx = numpy.column_stack((
                a_jetIdx,
                a_y,
                a_x,
                a_id
            ))
            
            for iConsti in range(0, nConsti) :
                
                #x  = a_x[iConsti]
                #y  = a_y[iConsti]
                #w  = numpy.float16(a_w[iConsti])
                #pdgid = abs(int(a_id[iConsti]))
                #
                #
                #bin_x = min(numpy.uint32((x - xMin) / xBinSize), nBinX-1)
                #bin_y = min(numpy.uint32((y - yMin) / yBinSize), nBinY-1)
                #
                #layer = pdgid2layer(pdgid)
                #
                #idx = (offset+jetIdx, bin_y, bin_x, layer)
                
                #if (idx in l_idx) :
                #    
                #    l_val[l_idx.index(idx)] += w
                #
                #else :
                #    
                #    l_idx.append(idx)
                #    l_val.append(w)
                
                #if (idx in d_arrayIdx.keys()) :
                #    
                #    d_arrayIdx.update({idx: w+d_arrayIdx.get(idx)})
                #
                #else :
                #    
                #    d_arrayIdx.update({idx: w})
                
                #d_arrayIdx.append(idx)
                
                idx = tuple(a_constiIdx[iConsti])
                
                if (idx in d_arrayIdx) :
                    
                    d_arrayIdx[idx] += a_w[iConsti]
                
                else :
                    
                    d_arrayIdx[idx] = a_w[iConsti]
            
            
            #del a_x
            #del a_y
            #del a_w
            #del a_id
            
            a_x  = None
            a_y  = None
            a_w  = None
            a_id = None
        
        #gc.collect()
        
        
        if (debugId is not None) :
            
            if (iEvent == 0 or iEvent+1 == nEvent or not (iEvent+1)%printEvery) :
                
                print("[%s] Processed event %d/%d." %(debugId, iEvent+1, nEvent))
    
    
    nJet_sel = jetIdx+1
    
    del branches
    #gc.collect()
    
    #return nJet_sel
    
    return (nJet_sel, d_arrayIdx)


#@memory_profiler.profile
def main() :
    
    l_branchName = [
        #"genTop_isLeptonic",
        #
        #"fatJet_pT_reco",
        #"fatJet_eta_reco",
        #"fatJet_m_reco",
        #"fatJet_nearestGenTopIdx_reco",
        #"fatJet_nearestGenTopDR_reco",
        #"fatJet_nearestGenTopIsLeptonic_reco",
        #"fatJet_nConsti_reco",
        "fatJet_constiTrans_x_reco",
        "fatJet_constiTrans_y_reco",
        "fatJet_constiTrans_w_reco",
        "fatJet_constiTrans_id_reco",
    ]
    
    print("=====> Loading trees... Memory:", getMemoryMB())
    
    d_catInfo = sortedcontainers.SortedDict()
    
    d_catInfo[0] = CategoryInfo(
        catNum = 0,
        l_fileAndTreeName = [
            "/nfs/dust/cms/user/sobhatta/work/TopTagPol/TreeMaker/CMSSW_10_5_0/src/ntupleTree_ZprimeToTT_M1000_W10.root:treeMaker/tree",
        ]*50,
        l_brName = l_branchName,
        cut = "(fatJet_pT_reco > 200) & (fatJet_eta_reco < 2.4) & (fatJet_nConsti_reco >= 3) & (fatJet_nearestGenTopDR_reco < 1) & (fatJet_nearestGenTopIsLeptonic_reco > 0.5)",
    )
    
    d_catInfo[1] = CategoryInfo(
        catNum = 1,
        l_fileAndTreeName = [
            "/nfs/dust/cms/user/sobhatta/work/TopTagPol/TreeMaker/CMSSW_10_5_0/src/ntupleTree_QCD_Pt_470to600.root:treeMaker/tree",
        ]*50,
        l_brName = l_branchName,
        cut = "(fatJet_pT_reco > 200) & (fatJet_eta_reco < 2.4) & (fatJet_nConsti_reco >= 3) & (fatJet_nearestGenTopDR_reco > 1)",
    )
    
    print("=====> Loaded trees... Memory:", getMemoryMB())
    
    
    jetIdx = 0
    
    l_idx = []
    l_val = []
    l_label = []
    
    d_sortedIdx = sortedcontainers.SortedDict()
    
    print("=====> Reading branches... Memory:", getMemoryMB())
    
    nEventPerJob = 5000
    
    
    #d_arrayIdx = multiprocessing.Manager().dict()
    
    #manager = MyManager()
    #manager.start()
    #sharedDict_arrayIdx = manager.SortedDict()
    
    sharedList_arrayIdx = multiprocessing.Manager().list()
    #sharedList_arrayIdx = multiprocessing.Manager().dict()
    
    
    imgSpec = ImageSpec(
        nBinX = 50,
        nBinY = 50,
        
        xMin = -1.0,
        xMax = +1.0,
        
        yMin = -1.0,
        yMax = +1.0,
    )
    
    
    #l_sparseArr_idx = []
    #l_sparseArr_val = []
    
    l_sparseArr_idx = collections.deque()
    l_sparseArr_val = collections.deque()
    
    for cat in d_catInfo :
        
        catInfo = d_catInfo[cat]
        catNum = catInfo.catNum
        
        nEventRead = 0
        
        #executor = concurrent.futures.ThreadPoolExecutor(max_workers = 10) 
        
        pool = multiprocessing.Pool(processes = 19)
        
        l_job = []
        
        for (iTree, tree) in enumerate(catInfo.l_tree) :
            
            nEvent = tree.num_entries
            #nEvent = 20000
            
            l_eventIdx = list(range(0, nEvent, nEventPerJob)) + [nEvent]
            
            for idx in range(0, len(l_eventIdx)-1) :
                
                entry_start = l_eventIdx[idx]
                entry_stop = l_eventIdx[idx+1]
                
                l_job.append(pool.apply_async(
                    computeBins,
                    (),
                    dict(
                        tree = tree,
                        l_brName = catInfo.l_brName,
                        cut = catInfo.cut,
                        entry_start = entry_start,
                        entry_stop = entry_stop,
                        #d_arrayIdx = sharedDict_arrayIdx,
                        #d_arrayIdx = sharedList_arrayIdx,
                        #l_label = ,
                        #offset = jetIdx,
                        offset = 0,
                        imgSpec = imgSpec,
                        debugId = "cat %d, tree %d/%d, job %d" %(catNum, iTree+1, len(catInfo.l_tree), len(l_job)+1),
                        printEvery = nEventPerJob,
                    ),
                ))
        
        print("\n%s Submitted all read-jobs for category %d %s\n" %("="*50, catNum, "="*50))
        
        pool.close()
        #pool.join()
        
        nJet_cat = 0
        
        l_isJobDone = [False] * len(l_job)
        
        while(False in l_isJobDone) :
            
            for iJob, job in enumerate(l_job) :
                
                #print iJob, job
                
                if (not l_isJobDone[iJob] and job.ready()) :
                    
                    l_isJobDone[iJob] = True
                    
                    nJet_sel, d_arrayIdx_temp = job.get()
                    
                    print("Finished read-job num %d (%d/%d done) of category %d." %(iJob+1, sum(l_isJobDone), len(l_job), catNum))
                    print("=====> Memory:", getMemoryMB())
                    
                    #nJet_sel = job.get()
                    
                    #d_arrayIdx_temp1 = [((jetIdx+keyIdx[0],)+keyIdx[1:], d_arrayIdx_temp[keyIdx]) for keyIdx in d_arrayIdx_temp]
                    
                    #print("*"*50, jetIdx, len(d_arrayIdx_temp.keys()), nJet_sel)
                    
                    l_sparseArr_idx.extend([(jetIdx+key[0],)+key[1:] for key in d_arrayIdx_temp.keys()])
                    l_sparseArr_val.extend(d_arrayIdx_temp.values())
                    
                    
                    
                    #for key in d_arrayIdx_temp.keys() :
                    #    
                    #    l_sparseArr_idx.append((jetIdx+key[0],)+key[1:])
                    #    l_sparseArr_val.append(d_arrayIdx_temp[key])
                    
                    jetIdx += nJet_sel
                    nJet_cat += nJet_sel
                    
                    #d_sortedIdx.update(d_arrayIdx_temp1)
                    
                    del d_arrayIdx_temp
                    #gc.collect()
                    
                    
                    if (not sum(l_isJobDone) % 10) :
                        
                        gc.collect()
                    
                    
                    
                    print("Processed output of read-job num %d (%d/%d done) of category %d." %(iJob+1, sum(l_isJobDone), len(l_job), catNum))
                    print("=====> Memory:", getMemoryMB())
        
        
        gc.collect()
        
        
        catInfo.nObj = nJet_cat
    
    
    # Fill the labels
    l_label = numpy.repeat(
        [d_catInfo[key].catNum for key in d_catInfo],
        [d_catInfo[key].nObj for key in d_catInfo],
    )
    
    print([d_catInfo[key].catNum for key in d_catInfo])
    print([d_catInfo[key].nObj for key in d_catInfo])
    
    
    print("=====> Read branches... Memory:", getMemoryMB())
    
    
    #l_label = numpy.array(l_label)
    
    nJet = len(l_label)
    print("nJet = %d (%d)" %(nJet, jetIdx))
    
    input_shape = (nJet, imgSpec.nBinY, imgSpec.nBinX, 8)
    img_shape = (imgSpec.nBinY, imgSpec.nBinX, 8)
    
    print("=====> Creating sparse tensor... Memory:", getMemoryMB())
    
    input_array = tensorflow.sparse.SparseTensor(
    #input_array = tensorflow.sparse.reorder(tensorflow.sparse.SparseTensor(
        #indices = l_idx,
        #values = l_val,
        #indices = d_sortedIdx.keys(),
        #values = d_sortedIdx.values(),
        indices = l_sparseArr_idx,
        values = l_sparseArr_val,
        dense_shape = input_shape,
    )
    
    #input_array = tensorflow.sparse.SparseTensor(
    ##input_array = tensorflow.sparse.reorder(tensorflow.sparse.SparseTensor(
    #    indices = sharedList_arrayIdx.keys(),
    #    values = sharedList_arrayIdx.values(),
    #    dense_shape = input_shape,
    #)
    
    print("=====> Created sparse tensor... Memory:", getMemoryMB())
    
    print(input_array)
    
    print("=====> Freeing dictionary... Memory:", getMemoryMB())
    #d_sortedIdx = None
    del d_sortedIdx
    del l_sparseArr_idx
    del l_sparseArr_val
    gc.collect()
    print("=====> Freed dictionary... Memory:", getMemoryMB())
    
    
    ##### Debug plot #####
    
    startIdx = 0
    #a_temp = tensorflow.make_ndarray(tensorflow.sparse.to_dense(input_array))
    arr_dense = tensorflow.sparse.to_dense(input_array).numpy()
    
    print(arr_dense.shape)
    
    for cat in d_catInfo :
        
        endIdx = startIdx + d_catInfo[cat].nObj
        
        a_temp = arr_dense[startIdx: endIdx] / d_catInfo[cat].nObj
        a_temp = a_temp.sum(axis = 0).sum(axis = -1)
        print(a_temp.shape)
        
        startIdx = endIdx
        
        
        fig = matplotlib.pyplot.figure(figsize = [10, 8])
        
        axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect("equal", "box")
        
        img = axis.imshow(
            a_temp,
            norm = matplotlib.colors.LogNorm(vmin = 1e-6, vmax = 1.0),
            origin = "lower",
            cmap = matplotlib.cm.get_cmap("nipy_spectral"),
        )
        
        fig.colorbar(img, ax = axis)
        
        fig.tight_layout()
        
        fig.savefig("jetImage_cat%d.pdf" %(d_catInfo[cat].catNum))
    
    
    
    #dataset_image = tensorflow.data.Dataset.from_tensor_slices(input_array)
    #dataset_label = tensorflow.data.Dataset.from_tensor_slices(l_label)
    ##print(input_dataset.element_spec)
    #
    #batch_size = 100
    #
    #print("=====> Creating dataset... Memory:", getMemoryMB())
    #dataset_train = tensorflow.data.Dataset.zip((dataset_image, dataset_label)).shuffle(buffer_size = nJet).batch(batch_size)
    #print("=====> Created dataset... Memory:", getMemoryMB())
    #print(dataset_train.element_spec)
    #
    #
    #model = models.Sequential()
    #
    #model.add(layers.Lambda(function = sparse_to_dense, input_shape = img_shape))
    #
    #model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    #model.add(layers.Flatten())
    #model.add(layers.Dense(64, activation = "relu"))
    #model.add(layers.Dense(10))
    #
    #model.summary()
    #
    #print("=====> Compiling model... Memory:", getMemoryMB())
    #
    #model.compile(
    #    optimizer = "adam",
    #    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    #    metrics = ["accuracy"],
    #    #run_eagerly = True,
    #    #shuffle = True,
    #)
    #
    #print("=====> Compiled model... Memory:", getMemoryMB())
    #
    ##history = model.fit(
    ##    train_images,
    ##    train_labels,
    ##    epochs = 5,
    ##    batch_size = batch_size,
    ##    validation_data = (test_images, test_labels),
    ##)
    #
    #print("=====> Starting fit... Memory:", getMemoryMB())
    #
    #history = model.fit(
    #    x = dataset_train,
    #    epochs = 5,
    #    #batch_size = batch_size,
    #    validation_data = dataset_train,
    #)



if (__name__ == "__main__") :
    
    main()
