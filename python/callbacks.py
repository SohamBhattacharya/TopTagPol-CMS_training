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
#import tensorboard.plugins.pr_curve.summary
import tensorboard.summary.v1
#import pr_curve.summary
#import my_pr_curve.summary
import time
import uproot
import yaml

from typing import Any, List, Set, Dict, Tuple, Optional


@dataclasses.dataclass
class CustomCallback(tensorflow.keras.callbacks.Callback):
    
    model               : Any
    writer              : Any
    writerPath          : Any
    dataset             : Any
    l_categoryLabel     : Any
    
    
    def __post_init__(
        self,
    ) :
        
        #self.d_cat_datasetIdx = {}
        #
        #d_data, a_label = self.dataset
        #
        #for cat in self.l_categoryLabel :
        #    
        #    self.d_cat_datasetIdx[cat] = numpy.where(a_label[:, cat] == 1)[0]
        
        self.prev_epoch = 0
        self.prev_step = 0
        
        self.curr_epoch = 0
        self.curr_step = 0
        
        print("Initialized instance of CustomCallback.")
        print(self.dataset[1].shape)
    
    
    def on_epoch_end(self, epoch, logs = None) :
        
        d_label = {}
        d_data, a_label = self.dataset
        
        #for cat in self.l_categoryLabel :
        #    
        #    d_pred[cat] = self.model.predict((
        #        {_key: tensorflow.gather(d_data[_key], indices = d_cat_datasetIdx[cat]) for _key in d_data.keys()},
        #        tensorflow.gather(a_label, indices = d_cat_datasetIdx[cat])
        #    ))
        
        a_pred = self.model.predict(d_data)
        
        for cat_sig in self.l_categoryLabel :
            
            #for iCat2 in range(iCat2, len(l_categoryLabel)) :
            for cat_bkg in self.l_categoryLabel :
                
                if (cat_sig == cat_bkg) :
                    
                    continue
                
                #pred_sig = d_pred[cat_sig][:, cat_sig] / (d_pred[cat_sig][:, cat_sig] + d_pred[cat_sig][:, cat_bkg])
                #pred_bkg = d_pred[cat_bkg][:, cat_sig] / (d_pred[cat_bkg][:, cat_sig] + d_pred[cat_bkg][:, cat_bkg])
                
                # sig (bkg) is column 1 (0)
                # Hence choose use "pos_label = 0" in sklearn.metrics.roc_curve
                nz_row, nz_col = numpy.nonzero(a_label[:, [cat_bkg, cat_sig]])
                
                a_pred_pair = a_pred[:, cat_sig][nz_row] / (a_pred[:, cat_sig][nz_row] + a_pred[:, cat_bkg][nz_row])
                
                #a_eff_bkg, a_eff_sig, a_threshold = sklearn.metrics.roc_curve(
                #    y_true = nz_col,
                #    y_score = a_pred_pair,
                #    pos_label = 1,
                #)
                #
                #auc = sklearn.metrics.auc(a_eff_bkg, a_eff_sig)
                
                cat_str = "sig-cat%d_bkg-cat%d" %(cat_sig, cat_bkg)
                
                print("Epoch %d, sig %d, bkg %d" %(epoch, cat_sig, cat_bkg), a_pred_pair)
                #print("Epoch %d, sig %d, bkg %d" %(epoch, cat_sig, cat_bkg), a_eff_bkg, a_eff_sig, a_threshold, auc)
                
                # https://github.com/tensorflow/tensorboard/issues/2902
                pr_curve_summary = tensorboard.summary.v1.pr_curve(
                #pr_curve_summary = pr_curve.summary.op(
                #pr_curve_summary = my_pr_curve.summary.op(
                    name = "PRC_%s" %(cat_str),
                    labels = nz_col.astype(bool),
                    predictions = a_pred_pair,
                )
                
                writer = tensorflow.summary.create_file_writer(self.writerPath)
                
                with writer.as_default():
                    
                    tensorflow.summary.experimental.write_raw_pb(pr_curve_summary, step = epoch)
                    
                    # [0] because where returns a tuple
                    tensorflow.summary.histogram(
                        "classifier_%s_sig" %(cat_str),
                        a_pred_pair[numpy.where(nz_col == 1)[0]],
                        step = epoch,
                        buckets = 500,
                    )
                    
                    tensorflow.summary.histogram(
                        "classifier_%s_bkg" %(cat_str),
                        a_pred_pair[numpy.where(nz_col == 0)[0]],
                        step = epoch,
                        buckets = 500,
                    )
                
    
    
    #def is_from_cat(self, x, y, catNum) :
    #    
    #    print(catNum, x, y, tensorflow.math.equal(y, catNum))
    #    #print(catNum, x)
    #    
    #    #return True
    #    return tensorflow.math.equal(y, catNum)
    #
    #
    #def on_train_batch_end(self, batch, logs = None) :
    #    
    #    self.prev_step = nBatch_trn*self.prev_epoch + batch
    #    self.curr_step = nBatch_trn*self.curr_epoch + batch
    #    
    #    #print("%"*10, self.prev_step)
    #    #print(self.model.predict(self.d_dataset_trn[0]))
    #    #print("on_train_batch_end logs:", logs)
    #    
    #    if (logs is None) :
    #        
    #        return
    #    
    #    if (
    #        not self.prev_step or
    #        not (self.prev_step%d_loadConfig["batchLog"]) or
    #        self.prev_step == nBatch_trn-1
    #    ) :
    #        
    #        with tensorboard_file_writer_trn.as_default() :
    #            
    #            tensorflow.summary.scalar("batch_loss", logs["loss"], step = self.curr_step)
    #            #tensorflow.summary.scalar("batch_accuracy", logs["accuracy"], step = self.curr_step)
    #            tensorflow.summary.scalar("batch_accuracy", logs["sparse_categorical_accuracy"], step = self.curr_step)
    #        
    #        with tensorboard_file_writer_tst.as_default() :
    #            
    #            y_pred = self.model.predict(self.cb_dataset_tst)
    #            
    #            accuracy_fn = tensorflow.keras.metrics.SparseCategoricalAccuracy()
    #            accuracy_fn.update_state(arr_label_tst, y_pred)
    #            accuracy = accuracy_fn.result().numpy()
    #            
    #            loss = loss_fn(arr_label_tst, y_pred).numpy()
    #            
    #            tensorflow.summary.scalar("batch_loss", loss, step = self.curr_step)
    #            tensorflow.summary.scalar("batch_accuracy", accuracy, step = self.curr_step)
    #
    #
    #def on_epoch_begin(self, epoch, logs = None) :
    #    
    #    self.curr_epoch = epoch
    #
    #
    #def on_epoch_end(self, epoch, logs = None) :
    #    
    #    self.prev_epoch = epoch
    #    
    #    with tensorboard_file_writer_trn.as_default() :
    #        
    #        y_pred = self.model.predict(self.cb_dataset_trn)
    #        
    #        accuracy_fn = tensorflow.keras.metrics.SparseCategoricalAccuracy()
    #        accuracy_fn.update_state(arr_label_trn, y_pred)
    #        accuracy = accuracy_fn.result().numpy()
    #        
    #        loss = loss_fn(arr_label_trn, y_pred).numpy()
    #        #print("my_epoch_loss (train):", loss)
    #        #
    #        #if (numpy.isnan(loss)) :
    #        #    
    #        #    print("Encountered nan in my_epoch_loss (train).")
    #        #    print(y_pred)
    #        
    #        tensorflow.summary.scalar("my_epoch_loss", loss, step = epoch)
    #        tensorflow.summary.scalar("my_epoch_accuracy", accuracy, step = epoch)
    #    
    #    
    #    with tensorboard_file_writer_tst.as_default() :
    #        
    #        y_pred = self.model.predict(self.cb_dataset_tst)
    #        
    #        accuracy_fn = tensorflow.keras.metrics.SparseCategoricalAccuracy()
    #        accuracy_fn.update_state(arr_label_tst, y_pred)
    #        accuracy = accuracy_fn.result().numpy()
    #        
    #        loss = loss_fn(arr_label_tst, y_pred).numpy()
    #        #print("my_epoch_loss (test):", loss)
    #        #
    #        #if (numpy.isnan(loss)) :
    #        #    
    #        #    print("Encountered nan in my_epoch_loss (test).")
    #        #    print(y_pred)
    #        
    #        tensorflow.summary.scalar("my_epoch_loss", loss, step = epoch)
    #        tensorflow.summary.scalar("my_epoch_accuracy", accuracy, step = epoch)
    #    
    #    
    #    
    #    with tensorboard_file_writer_params.as_default() :
    #        
    #        #tensorflow.summary.scalar("learning_rate", self.model.optimizer.learning_rate, step = epoch)
    #        tensorflow.summary.scalar("learning_rate", self.model.optimizer.learning_rate(self.prev_step), step = epoch)
    #    
    #    keys = list(logs.keys())
    #    #print(keys)
    #    #print("on_epoch_end logs:", logs)
    #    
    #    d_pred_trn = {}
    #    d_pred_tst = {}
    #    
    #    #print(
    #    #    "===== Epoch loss: train %0.4f, val %0.4f =====" %(
    #    #    loss_fn(
    #    #        numpy.array([ele for ele in dataset_label_trn.as_numpy_iterator()]),
    #    #        self.model.predict(
    #    #            tensorflow.data.Dataset.zip((dataset_image_trn, dataset_label_trn, dataset_weight_trn)).batch(batch_size = batch_size)
    #    #        )
    #    #    ).numpy(),
    #    #    
    #    #    loss_fn(
    #    #        numpy.array([ele for ele in dataset_label_tst.as_numpy_iterator()]),
    #    #        self.model.predict(
    #    #            tensorflow.data.Dataset.zip((dataset_image_tst, dataset_label_tst, dataset_weight_tst)).batch(batch_size = batch_size)
    #    #        )
    #    #    ).numpy(),
    #    #))
    #    
    #    #print(model.history.__dict__)
    #    
    #    #print("Trn pred shape", self.model.predict(dataset_trn).shape)
    #    #print("Tst pred shape", self.model.predict(dataset_tst).shape)
    #    
    #    for cat in d_catInfo_trn.keys() :
    #        
    #        pred_trn = self.model.predict(self.d_dataset_trn[cat])
    #        pred_tst = self.model.predict(self.d_dataset_tst[cat])
    #        
    #        d_pred_trn[cat] = pred_trn
    #        d_pred_tst[cat] = pred_tst
    #        
    #        print("Trn pred shape (cat%d)" %(cat), pred_trn.shape)
    #        print("Tst pred shape (cat%d)" %(cat), pred_tst.shape)
    #        
    #        #print(self.model.input)
    #        #print(self.model.output)
    #        #print("*"*10, cat)
    #        #print(pred_trn)
    #        
    #        for node in range(nCategory) :
    #            
    #            with tensorboard_file_writer_trn.as_default() :
    #                
    #                tensorflow.summary.histogram("output_node%d_cat%d" %(node, cat), pred_trn[:, node], step = epoch, buckets = 500)
    #            
    #            with tensorboard_file_writer_tst.as_default() :
    #                
    #                tensorflow.summary.histogram("output_node%d_cat%d" %(node, cat), pred_tst[:, node], step = epoch, buckets = 500)
    #    
    #    
    #    for iCat, cat_bkg in enumerate(range(1, nCategory)) :
    #        
    #        node_sig = 0
    #        node_bkg = cat_bkg
    #        cat_sig = 0
    #        
    #        pred_sig_trn = d_pred_trn[cat_sig][:, node_sig] / (d_pred_trn[cat_sig][:, node_sig] + d_pred_trn[cat_sig][:, node_bkg])
    #        pred_bkg_trn = d_pred_trn[cat_bkg][:, node_sig] / (d_pred_trn[cat_bkg][:, node_sig] + d_pred_trn[cat_bkg][:, node_bkg])
    #        
    #        pred_sig_tst = d_pred_tst[cat_sig][:, node_sig] / (d_pred_tst[cat_sig][:, node_sig] + d_pred_tst[cat_sig][:, node_bkg])
    #        pred_bkg_tst = d_pred_tst[cat_bkg][:, node_sig] / (d_pred_tst[cat_bkg][:, node_sig] + d_pred_tst[cat_bkg][:, node_bkg])
    #        
    #        with tensorboard_file_writer_trn.as_default() :
    #            
    #            tensorflow.summary.histogram("classifier_cat%dvs%d_cat%d" %(cat_sig, cat_bkg, cat_sig), pred_sig_trn, step = epoch, buckets = 500)
    #            tensorflow.summary.histogram("classifier_cat%dvs%d_cat%d" %(cat_sig, cat_bkg, cat_bkg), pred_bkg_trn, step = epoch, buckets = 500)
    #        
    #        with tensorboard_file_writer_tst.as_default() :
    #            
    #            tensorflow.summary.histogram("classifier_cat%dvs%d_cat%d" %(cat_sig, cat_bkg, cat_sig), pred_sig_tst, step = epoch, buckets = 500)
    #            tensorflow.summary.histogram("classifier_cat%dvs%d_cat%d" %(cat_sig, cat_bkg, cat_bkg), pred_bkg_tst, step = epoch, buckets = 500)
    #        
    #        
    #        # Train
    #        arr_eff_bkg_trn, arr_eff_sig_trn, arr_threshold_trn = sklearn.metrics.roc_curve(
    #            y_true = numpy.repeat(
    #                [cat_sig, cat_bkg],
    #                [len(pred_sig_trn), len(pred_bkg_trn)],
    #            ),
    #            y_score = numpy.concatenate(
    #                [
    #                    pred_sig_trn,
    #                    pred_bkg_trn,
    #                ],
    #                axis = None,
    #            ),
    #            pos_label = cat_sig,
    #        )
    #        
    #        auc_trn = sklearn.metrics.auc(arr_eff_bkg_trn, arr_eff_sig_trn)
    #        
    #        
    #        # Test
    #        arr_eff_bkg_tst, arr_eff_sig_tst, arr_threshold_tst = sklearn.metrics.roc_curve(
    #            y_true = numpy.repeat(
    #                [cat_sig, cat_bkg],
    #                [len(pred_sig_tst), len(pred_bkg_tst)],
    #            ),
    #            y_score = numpy.concatenate(
    #                [
    #                    pred_sig_tst,
    #                    pred_bkg_tst,
    #                ],
    #                axis = None,
    #            ),
    #            pos_label = cat_sig,
    #        )
    #        
    #        auc_tst = sklearn.metrics.auc(arr_eff_bkg_tst, arr_eff_sig_tst)
    #        
    #        
    #        with tensorboard_file_writer_trn.as_default() :
    #            
    #            tensorflow.summary.scalar("auc_cat%d_vs_cat%d" %(cat_sig, cat_bkg), auc_trn, step = epoch)
    #            
    #            tfimage = utils_tensorflow.get_tfimage_xyplot(
    #                l_plotdata = [
    #                    [
    #                        arr_eff_sig_trn, arr_eff_bkg_trn, "-", {}
    #                    ],
    #                ],
    #                xlim = (0, 1),
    #                ylim = (1e-6, 1),
    #                logy = True,
    #                xlabel = "cat%d efficiency" %(cat_sig),
    #                ylabel = "cat%d efficiency" %(cat_bkg),
    #            )
    #            tensorflow.summary.image("roc_cat%d_vs_cat%d" %(cat_sig, cat_bkg), tfimage, step = epoch)
    #            
    #            
    #            tfimage = utils_tensorflow.get_tfimage_xyplot(
    #                l_plotdata = [
    #                    [
    #                        arr_threshold_trn, arr_eff_sig_trn, "r-",
    #                        {
    #                            "label": "cat%d" %(cat_sig),
    #                        },
    #                    ],
    #                    [
    #                        arr_threshold_trn, arr_eff_bkg_trn, "b-",
    #                        {
    #                            "label": "cat%d" %(cat_bkg),
    #                        },
    #                    ],
    #                ],
    #                xlim = (0, 1),
    #                ylim = (1e-6, 1),
    #                logy = True,
    #                xlabel = "Cut on cat%d_vs_cat%d classifier" %(cat_sig, cat_bkg),
    #                ylabel = "Efficiency",
    #            )
    #            tensorflow.summary.image("eff_cat%d_vs_cat%d" %(cat_sig, cat_bkg), tfimage, step = epoch)
    #        
    #        with tensorboard_file_writer_tst.as_default() :
    #            
    #            tensorflow.summary.scalar("auc_cat%d_vs_cat%d" %(cat_sig, cat_bkg), auc_tst, step = epoch)
    #            
    #            tfimage = utils_tensorflow.get_tfimage_xyplot(
    #                l_plotdata = [
    #                    [
    #                        arr_eff_sig_tst, arr_eff_bkg_tst, "-", {}
    #                    ],
    #                ],
    #                xlim = (0, 1),
    #                ylim = (1e-6, 1),
    #                logy = True,
    #                xlabel = "cat%d efficiency" %(cat_sig),
    #                ylabel = "cat%d efficiency" %(cat_bkg),
    #            )
    #            tensorflow.summary.image("roc_cat%d_vs_cat%d" %(cat_sig, cat_bkg), tfimage, step = epoch)
    #            
    #            
    #            tfimage = utils_tensorflow.get_tfimage_xyplot(
    #                l_plotdata = [
    #                    [
    #                        arr_threshold_tst, arr_eff_sig_tst, "r-",
    #                        {
    #                            "label": "cat%d" %(cat_sig),
    #                        },
    #                    ],
    #                    [
    #                        arr_threshold_tst, arr_eff_bkg_tst, "b-",
    #                        {
    #                            "label": "cat%d" %(cat_bkg),
    #                        },
    #                    ],
    #                ],
    #                xlim = (0, 1),
    #                ylim = (1e-6, 1),
    #                logy = True,
    #                xlabel = "Cut on cat%d_vs_cat%d classifier" %(cat_sig, cat_bkg),
    #                ylabel = "Efficiency",
    #            )
    #            tensorflow.summary.image("eff_cat%d_vs_cat%d" %(cat_sig, cat_bkg), tfimage, step = epoch)
    #            



if (__name__ == "__main__") :
    
    main()
