�	�L�T4@�L�T4@!�L�T4@	w{���l�?w{���l�?!w{���l�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�L�T4@�wak�2�?1�,B�2@A��;�(A�?I�I�p�?Y�/���?*	�n���a@2F
Iterator::Model��(�֨?!	68%9A@)b.��n��?1�c9L�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice/���?!��G9�Z3@)/���?1��G9�Z3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat6<�R�!�?![����>@)P�D���?1ȱU�b�1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�wak��?!&��nd�)@)�wak��?1&��nd�)@:Preprocessing2U
Iterator::Model::ParallelMapV2�w�*�?!�^e��1)@)�w�*�?1�^e��1)@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��� �?!6�R,�>@)��X�?1]��2��&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�)��z��?!A��cmcP@)�A��x?1ɕ��б@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 6.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9w{���l�?I�7��� @Q�j��V@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�wak�2�?�wak�2�?!�wak�2�?      ��!       "	�,B�2@�,B�2@!�,B�2@*      ��!       2	��;�(A�?��;�(A�?!��;�(A�?:	�I�p�?�I�p�?!�I�p�?B      ��!       J	�/���?�/���?!�/���?R      ��!       Z	�/���?�/���?!�/���?b      ��!       JGPUYw{���l�?b q�7��� @y�j��V@�"K
$Adam/Adam/update_6/ResourceApplyAdamResourceApplyAdamm�FȀ'�?!m�FȀ'�?"5
sequential/dense/MatMulMatMul8�Q��H�?!����K�?0"C
'gradient_tape/sequential/dense/MatMul_1MatMul���8���?![����?"C
%gradient_tape/sequential/dense/MatMulMatMul���Ţ�?!�Ktlv��?0"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInputaYx"̇�?!�V��o��?0":
sequential/conv2d_2/Relu_FusedConv2D��W��W�?!U���V�?"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9&?�7�?!��lT���?0"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput�� ���?!��U���?0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�>��|�?!>��*�?0":
sequential/conv2d_1/Relu_FusedConv2Dv.���?!߿Ğ}�?Q      Y@Y/�袋.2@au�E]tT@q"
���@y�Z΢\j�?"�	
both�Your program is POTENTIALLY input-bound because 6.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 