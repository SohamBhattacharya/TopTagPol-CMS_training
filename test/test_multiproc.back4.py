from __future__ import print_function

#import mxnet
import awkward
import collections
import cppyy
import cppyy.ll
import concurrent.futures
import gc
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
import psutil
import pympler
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
        brName_nConsti,
        cut,
    ) :
        
        self.catNum = catNum
        self.l_fileAndTreeName = l_fileAndTreeName
        self.l_brName = l_brName
        self.brName_nConsti = brName_nConsti
        self.cut = cut
        
        #self.l_fileAndTreeName = l_fileAndTreeName
        
        #self.l_tree.append(uproot.open("%s:%s" %(fileName, fileAndTreeName), num_workers = 10))
        #self.tree = uproot.lazy("%s:%s" %(fileName, fileAndTreeName))
        
        #self.branches = self.tree.arrays(
        #    expressions = l_brName,
        #    cut = cut,
        #)
        
        #self.branches = uproot.lazy(
        #    files = [self.tree]
        #    expressions = l_brName,
        #    cut = cut,
        #)
        
        #self.l_tree = []
        #
        #for name in l_fileAndTreeName :
        #    
        #    self.l_tree.append(uproot.open(
        #        path = name,
        #    ))
        
        
        self.nJet = 0


class TrainingData :
    
    def __init__(
        self,
        nRow,
        nCol,
    ) :
        
        self.nRow = nRow
        self.nCol = nCol
        
        arr_idx = numpy.empty((nRow, 4), dtype = numpy.uint32)
        arr_val = numpy.empty(nRow, dtype = numpy.float16)
        
        self.arr_idx_shared = multiprocessing.shared_memory.SharedMemory(create = True, size = arr_idx.nbytes)
        self.arr_val_shared = multiprocessing.shared_memory.SharedMemory(create = True, size = arr_val.nbytes)
        
        self.arr_idx = numpy.ndarray(arr_idx.shape, dtype = arr_idx.dtype, buffer = self.arr_idx_shared.buf)
        self.arr_val = numpy.ndarray(arr_val.shape, dtype = arr_val.dtype, buffer = self.arr_val_shared.buf)
        
        self.arr_idx_shape = arr_idx.shape
        self.arr_val_shape = arr_val.shape
        
        self.arr_idx_dtype = arr_idx.dtype
        self.arr_val_dtype = arr_val.dtype
        
    
    #def fill(self, result) :
    #    
    #    fill_start, fill_end, arr_idx_temp, arr_val_temp = result
    #    
    #    self.arr_idx[fill_start: fill_end] = arr_idx_temp
    #    self.arr_val[fill_start: fill_end] = arr_val_temp
    #    
    #    result = None
    #    fill_start = None
    #    fill_end = None
    #    arr_idx_temp = None
    #    arr_val_temp = None
    #    
    #    del result
    #    del fill_start
    #    del fill_end
    #    del arr_idx_temp
    #    del arr_val_temp
    #    
    #    #cppyy.ll.free(arr_idx_temp)
    #    #cppyy.ll.free(arr_val_temp)
    #    
    #    print("size(arr_idx) in MB:", self.arr_idx.nbytes/1024.0**2)
    #    print("size(arr_val) in MB:", self.arr_val.nbytes/1024.0**2)
    #    
    #    gc.collect()


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
    fileAndTreeName,
    l_brName,
    cut,
    entry_start,
    entry_stop,
    jetIdxOffset,
    nConsti,
    imgSpec,
    arr_idx_shared,
    arr_val_shared,
    arr_idx_shape,
    arr_val_shape,
    arr_idx_dtype,
    arr_val_dtype,
    fill_start,
    fill_end,
    debugId = None,
    printEvery = 1000,
) :
    
    if (entry_start >=  entry_stop) :
        
        return 0
    
    
    np_idx_shared = numpy.ndarray(arr_idx_shape, dtype = arr_idx_dtype, buffer = arr_idx_shared.buf)
    np_val_shared = numpy.ndarray(arr_val_shape, dtype = arr_val_dtype, buffer = arr_val_shared.buf)
    
    
    arr_idx = numpy.zeros((nConsti, 4), dtype = numpy.uint32)
    arr_val = numpy.zeros(nConsti, dtype = numpy.float16)
    
    
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
    
    
    #d_arrayIdx = sortedcontainers.SortedDict()
    #d_arrayIdx = {}
    
    entryCount = 0
    
    with uproot.open(fileAndTreeName) as tree :
        
        branches = tree.arrays(
            expressions = l_brName,
            cut = cut,
            entry_start = entry_start,
            entry_stop = entry_stop,
        )
        
        nEvent = len(branches[l_brName[0]])
        #nEvent = 1000
        
        
        for iEvent in range(0, nEvent) :
            
            nJet = len(branches[l_brName[0]][iEvent])
            
            for iJet in range(0, nJet) :
                
                jetIdx += 1
                
                a_x  = ((branches["fatJet_constiTrans_x_reco"][iEvent][iJet].to_numpy() - xMin) / xBinSize).astype(dtype = numpy.uint32)
                a_y  = ((branches["fatJet_constiTrans_y_reco"][iEvent][iJet].to_numpy() - yMin) / yBinSize).astype(dtype = numpy.uint32)
                a_w  = branches["fatJet_constiTrans_w_reco"][iEvent][iJet].to_numpy().astype(dtype = numpy.float16, copy = False)
                
                #print(branches["fatJet_constiTrans_id_reco"][iEvent][iJet])
                a_id = vec_pdgid2layer(branches["fatJet_constiTrans_id_reco"][iEvent][iJet].to_numpy())
                
                jet_nConsti = len(a_x)
                
                a_jetIdx = numpy.full(jet_nConsti, fill_value = jetIdxOffset+jetIdx, dtype = numpy.uint32)
                
                
                a_constiIdx = numpy.column_stack((
                    a_jetIdx,
                    a_y,
                    a_x,
                    a_id
                ))
                
                d_arrayIdx = sortedcontainers.SortedDict()
                
                for iConsti in range(0, jet_nConsti) :
                    
                    idx = tuple(a_constiIdx[iConsti])
                    
                    if (idx in d_arrayIdx) :
                        
                        d_arrayIdx[idx] += a_w[iConsti]
                    
                    else :
                        
                        d_arrayIdx[idx] = a_w[iConsti]
                
                for key in d_arrayIdx.keys() :
                    
                    arr_idx[entryCount] = key
                    arr_val[entryCount] = d_arrayIdx[key]
                    
                    entryCount += 1
                
                
                a_x  = None
                a_y  = None
                a_w  = None
                a_id = None
            
            #gc.collect()
            
            
            if (debugId is not None) :
                
                if (iEvent == 0 or iEvent+1 == nEvent or not (iEvent+1)%printEvery) :
                    
                    print("[%s] Processed event %d/%d." %(debugId, iEvent+1, nEvent))
    
    
    #print("%"*10, nJet_sel, len(d_arrayIdx))
    
    #return (arr_idx[0: entryCount], arr_val[0: entryCount])
    #return (fill_start, fill_end, arr_idx, arr_val)
    
    #return (nJet_sel, d_arrayIdx)
    
    
    np_idx_shared[fill_start: fill_end] = arr_idx_temp
    np_val_shared[fill_start: fill_end] = arr_val_temp
    
    return 0


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
        ]*1,
        l_brName = l_branchName,
        brName_nConsti = "fatJet_nConsti_reco",
        cut = "(fatJet_pT_reco > 200) & (fatJet_eta_reco < 2.4) & (fatJet_nConsti_reco >= 3) & (fatJet_nearestGenTopDR_reco < 1) & (fatJet_nearestGenTopIsLeptonic_reco > 0.5)",
    )
    
    d_catInfo[1] = CategoryInfo(
        catNum = 1,
        l_fileAndTreeName = [
            "/nfs/dust/cms/user/sobhatta/work/TopTagPol/TreeMaker/CMSSW_10_5_0/src/ntupleTree_QCD_Pt_470to600.root:treeMaker/tree",
        ]*1,
        l_brName = l_branchName,
        brName_nConsti = "fatJet_nConsti_reco",
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
    
    
    l_sparseArr_idx = []
    l_sparseArr_val = []
    
    #l_sparseArr_idx = collections.deque()
    #l_sparseArr_val = collections.deque()
    
    nJetTot = 0
    nConstiTot = 0
    
    for (iCat, cat) in enumerate(d_catInfo.keys()) :
        
        catInfo = d_catInfo[cat]
        brName_nConsti = catInfo.brName_nConsti
        
        for (iTree, fileAndTreeName) in enumerate(catInfo.l_fileAndTreeName) :
            
            nEvent = 0
            
            with uproot.open(fileAndTreeName) as tree :
                
                nEvent = tree.num_entries
                
                l_eventIdx = list(range(0, nEvent, nEventPerJob))
                
                if (l_eventIdx[-1] != nEvent) :
                    
                    l_eventIdx.append(nEvent)
                
                for idx in range(0, len(l_eventIdx)-1) :
                    
                    entry_start = l_eventIdx[idx]
                    entry_stop = l_eventIdx[idx+1]
                    
                    branches = tree.arrays(
                        expressions = [brName_nConsti],
                        cut = catInfo.cut,
                        entry_start = entry_start,
                        entry_stop = entry_stop,
                    )
                    
                    a_nConsti = awkward.values_astype(branches[brName_nConsti], to = numpy.int64)
                    
                    nConsti = awkward.sum(a_nConsti)
                    
                    nConstiTot += nConsti
    
    print("nConstiTot:", nConstiTot)
    
    l_job = []
    l_catNum = []
    l_nJet = []
    l_nConsti = []
    l_constiIdx = []
    
    trnData = TrainingData(
        nRow = nConstiTot,
        nCol = 4,
    )
    
    nJetTot = 0
    nConstiTot = 0
    
    pool = multiprocessing.Pool(processes = 19, maxtasksperchild = 1)
    
    for (iCat, cat) in enumerate(d_catInfo.keys()) :
        
        catInfo = d_catInfo[cat]
        catNum = catInfo.catNum
        
        brName_nConsti = catInfo.brName_nConsti
        
        #pool = multiprocessing.Pool(processes = 19)#, maxtasksperchild = 1)
        
        for (iTree, fileAndTreeName) in enumerate(catInfo.l_fileAndTreeName) :
            
            nEvent = 0
            
            with uproot.open(fileAndTreeName) as tree :
                
                nEvent = tree.num_entries
                
                l_eventIdx = list(range(0, nEvent, nEventPerJob))
                
                if (l_eventIdx[-1] != nEvent) :
                    
                    l_eventIdx.append(nEvent)
                
                for idx in range(0, len(l_eventIdx)-1) :
                    
                    entry_start = l_eventIdx[idx]
                    entry_stop = l_eventIdx[idx+1]
                    
                    branches = tree.arrays(
                        expressions = [brName_nConsti],
                        cut = catInfo.cut,
                        entry_start = entry_start,
                        entry_stop = entry_stop,
                    )
                    
                    a_nConsti = awkward.values_astype(branches[brName_nConsti], to = numpy.int64)
                    
                    nJet = awkward.count(a_nConsti)
                    nConsti = awkward.sum(a_nConsti)
                    
                    fill_start = nConstiTot
                    fill_end = nConstiTot + nConsti
                    
                    l_job.append(pool.apply_async(
                        computeBins,
                        (),
                        dict(
                            fileAndTreeName = fileAndTreeName,
                            l_brName = catInfo.l_brName,
                            cut = catInfo.cut,
                            entry_start = entry_start,
                            entry_stop = entry_stop,
                            jetIdxOffset = nJetTot,
                            nConsti = nConsti,
                            imgSpec = imgSpec,
                            arr_idx_shared = trnData.arr_idx,
                            arr_val_shared = trnData.arr_val,
                            fill_start = fill_start,
                            fill_end = fill_end,
                            debugId = "cat %d, tree %d/%d, job %d" %(catNum, iTree+1, len(catInfo.l_fileAndTreeName), len(l_job)+1),
                            printEvery = nEventPerJob,
                        ),
                        #callback = trnData.fill,
                    ))
                    
                    l_catNum.append(catNum)
                    l_nJet.append(nJet)
                    l_nConsti.append(nConsti)
                    l_constiIdx.append(nConstiTot)
                    
                    catInfo.nJet += nJet
                    nJetTot += nJet
                    nConstiTot += nConsti
        
        
        print("\n%s Submitted all read-jobs for category %d %s\n" %("="*50, catNum, "="*50))
        
        #pool.close()
        #pool.join()
    
    
    
    print("nJetTot:", nJetTot)
    print("nConstiTot:", nConstiTot)
    
    pool.close()
    #pool.join()
    
    arr_idx = trnData.arr_idx
    arr_val = trnData.arr_val
    
    #arr_idx = numpy.zeros((nConstiTot, 4), dtype = numpy.int64)
    #arr_val = numpy.zeros(nConstiTot, dtype = numpy.float16)
    #
    ##time.sleep(10)
    ##exit()
    
    
    l_isJobDone = [False] * len(l_job)
    
    while(False in l_isJobDone) :
        
        for iJob, job in enumerate(l_job) :
            
            #print iJob, job
            
            if (job is None) :
                
                continue
            
            if (not l_isJobDone[iJob] and job.ready()) :
                
                l_isJobDone[iJob] = True
                
                #arr_idx_temp, arr_val_temp = job.get()
                #
                #print("Finished read-job num %d (%d/%d done) of category %d." %(iJob+1, sum(l_isJobDone), len(l_job), catNum))
                #print("=====> Memory:", getMemoryMB())
                #
                #fill_start = l_constiIdx[iJob]
                #fill_end = l_constiIdx[iJob] + l_nConsti[iJob]
                #
                #arr_idx[fill_start: fill_end] = arr_idx_temp
                #arr_val[fill_start: fill_end] = arr_val_temp
                
                l_job[iJob] = None
                
                if (not sum(l_isJobDone) % 20) :
                    
                    gc.collect()
                
                print("Processed output of read-job num %d (%d/%d done) of category %d." %(iJob+1, sum(l_isJobDone), len(l_job), catNum))
                print("=====> Memory:", getMemoryMB())
    
    
    pool.join()
    
    
    # Fill the labels
    l_label = numpy.repeat(
        [d_catInfo[key].catNum for key in d_catInfo],
        [d_catInfo[key].nJet for key in d_catInfo],
    )
    
    print([d_catInfo[key].catNum for key in d_catInfo])
    print([d_catInfo[key].nJet for key in d_catInfo])
    
    
    print("=====> Read branches... Memory:", getMemoryMB())
    
    
    #l_label = numpy.array(l_label)
    
    nJetTot = len(l_label)
    print("nJetTot = %d (%d)" %(nJetTot, jetIdx))
    
    input_shape = (nJetTot, imgSpec.nBinY, imgSpec.nBinX, 8)
    img_shape = (imgSpec.nBinY, imgSpec.nBinX, 8)
    
    print("=====> Creating sparse tensor... Memory:", getMemoryMB())
    
    arr_nonzero_idx = (arr_idx != [0, 0, 0, 0]).any(axis = 1).nonzero()
    
    input_array = tensorflow.sparse.SparseTensor(
        indices = arr_idx[arr_nonzero_idx],
        values = arr_val[arr_nonzero_idx],
        dense_shape = input_shape,
    )
    
    print("=====> Created sparse tensor... Memory:", getMemoryMB())
    
    print(input_array)
    
    #print("=====> Freeing dictionary... Memory:", getMemoryMB())
    ##d_sortedIdx = None
    #del d_sortedIdx
    #del l_sparseArr_idx
    #del l_sparseArr_val
    #del arr_idx
    #del arr_val
    #del arr_nonzero_idx
    #del trnData
    #gc.collect()
    #print("=====> Freed dictionary... Memory:", getMemoryMB())
    #
    #
    #print("Waiting...")
    #time.sleep(50)
    
    
    ##### Debug plot #####
    
    startIdx = 0
    #a_temp = tensorflow.make_ndarray(tensorflow.sparse.to_dense(input_array))
    arr_dense = tensorflow.sparse.to_dense(input_array).numpy()
    
    print(arr_dense.shape)
    
    for cat in d_catInfo :
        
        endIdx = startIdx + d_catInfo[cat].nJet
        
        a_temp = arr_dense[startIdx: endIdx] / d_catInfo[cat].nJet
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
