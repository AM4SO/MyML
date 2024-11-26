�	p#e���&@p#e���&@!p#e���&@	-$�Ƀ`�?-$�Ƀ`�?!-$�Ƀ`�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6p#e���&@��0�?1�����"@A�N�z1��?I�?����?Y.�R���?*��(\�]@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��8*7Q�?!��Y�F@)���ͧ?1�|T��C@:Preprocessing2U
Iterator::Model::ParallelMapV2����?!rˣ���/@)����?1rˣ���/@:Preprocessing2F
Iterator::Model�U�3��?!��7�=@)�N��唐?1�H_���+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��.ޏ�?![f�H�'@)��.ޏ�?1[f�H�'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"ߥ�%�?!*aY��4@)3�Vzm6�?1��0j"�"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorX�\T|?!���M�@)X�\T|?1���M�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���E�?!|����Q@)��25	�p?1dt{L@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 12.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9-$�Ƀ`�?I��@���-@Q������T@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��0�?��0�?!��0�?      ��!       "	�����"@�����"@!�����"@*      ��!       2	�N�z1��?�N�z1��?!�N�z1��?:	�?����?�?����?!�?����?B      ��!       J	.�R���?.�R���?!.�R���?R      ��!       Z	.�R���?.�R���?!.�R���?b      ��!       JGPUY-$�Ƀ`�?b q��@���-@y������T@�"K
$Adam/Adam/update_6/ResourceApplyAdamResourceApplyAdamr�A�?!r�A�?"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput���L�?!���F��?0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter"m� ��?!f 3iq��?0":
sequential/conv2d_2/Relu_FusedConv2D�St��?!��#ô_�?"C
%gradient_tape/sequential/dense/MatMulMatMul�T1cF�?!�vN)�h�?0"5
sequential/dense/MatMulMatMul��z,��?!�!�=I�?0"C
'gradient_tape/sequential/dense/MatMul_1MatMulX���?!J��|���?"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput�(3A��?!ҹ�P-��?0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilters�ޯ
��?!y���=��?0":
sequential/conv2d_1/Relu_FusedConv2De&w��?�?!�x�8��?Q      Y@Y]t�E]0@a袋.��T@q�*�`�$@yw�+��?"�

both�Your program is POTENTIALLY input-bound because 12.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�10.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 