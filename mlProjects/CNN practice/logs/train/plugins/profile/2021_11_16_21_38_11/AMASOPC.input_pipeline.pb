	]5���%@]5���%@!]5���%@	����P
@����P
@!����P
@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6]5���%@3��bb��?1ZEh�Y!@AMN�S[�?I��-�l�?Y�rJ@L��?*	�Zd;_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�0ҋ���?!����,E@)�҇.�o�?1��+&[;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapY��9�}�?!M9��͔>@)Y�U���?1>2�fa6@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor仔�d�?!�o���-@)仔�d�?1�o���-@:Preprocessing2U
Iterator::Model::ParallelMapV2+����?!�� ��*@)+����?1�� ��*@:Preprocessing2F
Iterator::Model;Qi�?!p6�� g8@)!�> �M�?1?���3&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����t�?!m��f @)����t�?1m��f @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��9̗�?!d2A�?�R@)�o+�6k?1���i&Y@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 12.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����P
@I���u0@Q����T@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	3��bb��?3��bb��?!3��bb��?      ��!       "	ZEh�Y!@ZEh�Y!@!ZEh�Y!@*      ��!       2	MN�S[�?MN�S[�?!MN�S[�?:	��-�l�?��-�l�?!��-�l�?B      ��!       J	�rJ@L��?�rJ@L��?!�rJ@L��?R      ��!       Z	�rJ@L��?�rJ@L��?!�rJ@L��?b      ��!       JGPUY����P
@b q���u0@y����T@