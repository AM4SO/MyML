	G6u@G6u@!G6u@	�E�ג@�E�ג@!�E�ג@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6G6u@���M��?1>$|�o�@A��x!�?I�x�Z���?Y��YKi�?*	fffff�W@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat,��A$�?!\d�4zC@)*:��H�?1��S���?@:Preprocessing2U
Iterator::Model::ParallelMapV2:=�Ƃ�?!�(f���2@):=�Ƃ�?1�(f���2@:Preprocessing2F
Iterator::ModelC�+j�?!�2��ǳ@@)ӈ�}��?1�y �#-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap؁sF���?!�Y@Ŏ]7@)��_Z�'�?1�צƣ�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�;�2Tń?!S���y"%@)�;�2Tń?1S���y"%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���� |?!�P���}@)���� |?1�P���}@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�M�»\�?!�f$�P@)�H�[�p?1����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�13.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�E�ג@I��P���@@Q�x[��N@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���M��?���M��?!���M��?      ��!       "	>$|�o�@>$|�o�@!>$|�o�@*      ��!       2	��x!�?��x!�?!��x!�?:	�x�Z���?�x�Z���?!�x�Z���?B      ��!       J	��YKi�?��YKi�?!��YKi�?R      ��!       Z	��YKi�?��YKi�?!��YKi�?b      ��!       JGPUY�E�ג@b q��P���@@y�x[��N@