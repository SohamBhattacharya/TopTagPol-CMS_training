import numpy
import tensorflow
import tensorboard
#import tensorboard.plugins.pr_curve.summary
import tensorboard.summary.v1
#import my_pr_curve.summary
import os

nbatch = 1
batchsize = 500
nclass = 2

#def get_arr() :
#    
#    a_pred = numpy.zeros((batchsize, nclass))
#    a_label = numpy.zeros((batchsize, nclass))
#    
#    for i in range(0, batchsize) :
#        
#        a_pred[0] = numpy.random.random_sample((nclass,))
#        a_pred[0] /= sum(a_pred[0])
#        
#        idx = numpy.random.randint(low = 0, high = nclass)
#        a_label[i, idx] = 1
#    
#    
#    return (a_pred, a_label)


def get_arr() :
    
    a_pred = numpy.zeros(batchsize, dtype = numpy.float32)
    a_label = numpy.zeros(batchsize)
    
    for i in range(0, batchsize) :
        
        a_pred[i] = numpy.random.random_sample()
        
        a_label[i] = numpy.random.randint(low = 0, high = 2)
    
    return (a_pred, a_label)



#writer = tensorflow.summary.create_file_writer("tmp/logdir")

a_pred, a_label = get_arr()

pr_curve_summary = tensorboard.summary.v1.pr_curve(
#pr_curve_summary = my_pr_curve.summary.op(
    name = "PRC",
    labels = a_label.astype(bool),
    predictions = a_pred,
    #predictions = tensorflow.constant(a_pred),
)

writer_dir = "tmp/my_plugin_logdir"
os.system("rm -r %s" %(writer_dir))

writer = tensorflow.summary.create_file_writer(writer_dir)

with writer.as_default():
    
    tensorflow.summary.experimental.write_raw_pb(pr_curve_summary, step = 0)

#with tensorflow.compat.v1.Session() as sess:
#    
#    
#    
#    writer_trn = tensorflow.compat.v1.summary.FileWriter("tmp/logdir/trn")
#    writer_tst = tensorflow.compat.v1.summary.FileWriter("tmp/logdir/tst")
#    
#    
#    
#    for step in range(10):
#        
#        print("step %d" %(step))
#        
#        a_pred, a_label = get_arr()
#        
#        for writer in [writer_trn, writer_tst] :
#            
#            pr_curve, update_op = tensorboard.plugins.pr_curve.summary.streaming_op(
#                name = "foo",
#                predictions = a_pred,
#                labels = a_label,
#                num_thresholds = 11
#            )
#            
#            merged_summary = tensorflow.compat.v1.summary.merge_all()
#            sess.run(tensorflow.compat.v1.local_variables_initializer())
#            
#            sess.run([update_op])
#            writer.add_summary(sess.run(merged_summary), global_step = step)
#


#writer = tensorflow.summary.create_file_writer("tmp/logdir")
#tensorflow.compat.v1.disable_v2_behavior()
##tensorflow.compat.v1.enable_v2_behavior()
##tensorflow.compat.v1.enable_eager_execution()
#
#
##with writer.as_default(step = 0) :
#
#with tensorflow.compat.v1.Session() as sess:
#    
#    pr_curve, update_op = tensorboard.plugins.pr_curve.summary.streaming_op(
#        name = "foo",
#        predictions = a_pred,
#        labels = a_label,
#        num_thresholds = 11
#    )
#    
#    merged_summary = tensorflow.compat.v1.summary.merge_all()
#    
#    #writer = tensorflow.summary.create_file_writer("tmp/logdir")
#    
#    sess.run(tensorflow.compat.v1.local_variables_initializer())
#    
#    for step in range(10):
#        
#        print("step %d" %(step))
#        
#        #sess.run([pr_curve, update_op])
#        #writer.add_summary(sess.run(merged_summary), global_step = step)
#        
#        with writer.as_default() :
#            
#            sess.run([pr_curve, update_op])
#            tensorflow.summary.experimental.write_raw_pb(pr_curve, step = step)
