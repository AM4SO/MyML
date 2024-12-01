�	DkE��!@DkE��!@!DkE��!@	��)��@��)��@!��)��@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6DkE��!@mU��?1��^~��@A����Li�?I���ڧ��?Y�r����?*	�� �r Z@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�&k�C4�?!�Rc)��H@)�IӠh�?1L��d�F@:Preprocessing2U
Iterator::Model::ParallelMapV2�F�0}��?!���u	�0@)�F�0}��?1���u	�0@:Preprocessing2F
Iterator::Model����ӟ?!���Y1�=@)F���jH�?1��n�O�*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]P�2�˒?!����٥1@)�o�^}<�?1Eܫ7* #@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�0��Z�?!G܈K @)�0��Z�?1G܈K @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor辜ٮp?!v�c#T@)辜ٮp?1v�c#T@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipM1AG��?!���s�Q@)�:9Cq�k?1�k�L6
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��)��@I��9j�3@Q�l�4�zS@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	mU��?mU��?!mU��?      ��!       "	��^~��@��^~��@!��^~��@*      ��!       2	����Li�?����Li�?!����Li�?:	���ڧ��?���ڧ��?!���ڧ��?B      ��!       J	�r����?�r����?!�r����?R      ��!       Z	�r����?�r����?!�r����?b      ��!       JGPUY��)��@b q��9j�3@y�l�4�zS@�"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterky�����?!ky�����?0":
sequential/conv2d_1/Relu_FusedConv2D���O��?!���K��?"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput�C-�p�?!��k�u'�?0"K
$Adam/Adam/update_6/ResourceApplyAdamResourceApplyAdam��k$�y�?!qdÄ��?"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput�d:,�}�?!�
�����?0"C
%gradient_tape/sequential/dense/MatMulMatMulԜ�b�ԩ?!����h�?0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter> 'a��?!�� ���?0":
sequential/conv2d_2/Relu_FusedConv2Dy����?!B,�����?"5
sequential/dense/MatMulMatMul�O>���?!�$����?0"C
'gradient_tape/sequential/dense/MatMul_1MatMul7�bOʠ?!s ��9�?Q      Y@Y]t�E]0@a袋.��T@q����!@y���?��?"�

both�Your program is POTENTIALLY input-bound because 14.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 