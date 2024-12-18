�	�[Z��@�[Z��@!�[Z��@	A9� �?A9� �?!A9� �?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�[Z��@���~�?1}�ݮ��@A�'��?I��}�[.�?Y��x>��?*		�Zd�Z@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_z�sѐ�?!8���PYG@)�]~p�?1�b~D@:Preprocessing2U
Iterator::Model::ParallelMapV2���N�?!�Uai��.@)���N�?1�Uai��.@:Preprocessing2F
Iterator::Model���O��?!�򇙈V;@)Lk��^�?1����=�'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�
(�ӗ?!���H��5@)UQ��ڦ�?1�e���&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�pY�� �?!3p��.%@)�pY�� �?13p��.%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@4��y?!��o�@)@4��y?1��o�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!����?!K��]*R@)�s�Lhr?1Ar)��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 13.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9C9� �?IW Q~�5@QV����S@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���~�?���~�?!���~�?      ��!       "	}�ݮ��@}�ݮ��@!}�ݮ��@*      ��!       2	�'��?�'��?!�'��?:	��}�[.�?��}�[.�?!��}�[.�?B      ��!       J	��x>��?��x>��?!��x>��?R      ��!       Z	��x>��?��x>��?!��x>��?b      ��!       JGPUYC9� �?b qW Q~�5@yV����S@�"K
$Adam/Adam/update_6/ResourceApplyAdamResourceApplyAdamT�i���?!T�i���?"C
%gradient_tape/sequential/dense/MatMulMatMul�?\U��?!��b^�?0"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput-V2��?!���1)�?0"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput�.p蠳?!l�l�?0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltero���q�?!a��O���?0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter���b��?!�耒�?0":
sequential/conv2d_2/Relu_FusedConv2D�>"��?!�Uɤ���?"5
sequential/dense/MatMulMatMul�m%Q��?!���iV��?0":
sequential/conv2d_1/Relu_FusedConv2D�lo^~8�?!j��O�H�?"C
'gradient_tape/sequential/dense/MatMul_1MatMulm�b�,c�?!!Θ��?Q      Y@Y%I�$I0@a�m۶m�T@q�Ռx�(@yZC&�?"�
both�Your program is POTENTIALLY input-bound because 13.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�12.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 