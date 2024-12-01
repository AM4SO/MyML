�	�e�YJ�+@�e�YJ�+@!�e�YJ�+@	Aɋ��~@Aɋ��~@!Aɋ��~@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�e�YJ�+@�\��u�?1�`p��'@A+��$Ί�?I���?Y[\�3�?�?*⥛� 8^@)      =2U
Iterator::Model::ParallelMapV2Â��?!I��>��9@)Â��?1I��>��9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ek}�Ц?!Y�5�nB@)J`s�	�?1T�-6�u7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatd]�Fx�?!H8��%16@)�ѩ+��?1�H��w1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	kc섗�?!��z<�*@)	kc섗�?1��z<�*@:Preprocessing2F
Iterator::Model���!��?!��a?C@)3SZK �?1.���<(@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��)t^cw?!�1}.�@)��)t^cw?1�1}.�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/�혺+�?!s����N@)7 !�l?13��q�+@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Aɋ��~@I }b��x'@QRO��TU@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�\��u�?�\��u�?!�\��u�?      ��!       "	�`p��'@�`p��'@!�`p��'@*      ��!       2	+��$Ί�?+��$Ί�?!+��$Ί�?:	���?���?!���?B      ��!       J	[\�3�?�?[\�3�?�?![\�3�?�?R      ��!       Z	[\�3�?�?[\�3�?�?![\�3�?�?b      ��!       JGPUYAɋ��~@b q }b��x'@yRO��TU@�"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput��Z�?!��Z�?0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�Ϳ�=�?!���?�|�?0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�o�!J'�?!��HȦF�?0":
sequential/conv2d_1/Relu_FusedConv2D��IS9��?!0*u��?"K
$Adam/Adam/update_6/ResourceApplyAdamResourceApplyAdam��Ac�Q�?!�������?"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput��}���?!�����?0":
sequential/conv2d_2/Relu_FusedConv2D�jN���?!�n�]
��?"C
%gradient_tape/sequential/dense/MatMulMatMulB�_6�?!4��k��?0"5
sequential/dense/MatMulMatMul����֡?!pڔ���?0"C
'gradient_tape/sequential/dense/MatMul_1MatMul��cƅc�?!��[	�?Q      Y@Y������0@aVUUUU�T@q����@y^#�?/��?"�	
both�Your program is POTENTIALLY input-bound because 9.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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