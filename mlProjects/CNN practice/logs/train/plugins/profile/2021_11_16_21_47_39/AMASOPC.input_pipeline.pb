	ӿ$�),@ӿ$�),@!ӿ$�),@	�WÜ��@�WÜ��@!�WÜ��@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ӿ$�),@������?1KVE�� (@A���Д��?I��[1��?Y�����?*	�A`��:^@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���FXT�?!�xu'k@@)��;��?1X9����5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�]=�1�?!�쵚T�3@)�]=�1�?1�쵚T�3@:Preprocessing2U
Iterator::Model::ParallelMapV2�@I�0�?!`��L�1@)�@I�0�?1`��L�1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat`w���s�?!�xJ(�!B@):�`����?1�ߵ��0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceC,cC7�?!Fqg|��%@)C,cC7�?1Fqg|��%@:Preprocessing2F
Iterator::Model��q��Q�?!�=g���;@)k�3�?1����%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZiptD�K�K�?!�0�ҚR@)���U�l?1]��X�a@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�WÜ��@I���'@Q�b�Á]U@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������?������?!������?      ��!       "	KVE�� (@KVE�� (@!KVE�� (@*      ��!       2	���Д��?���Д��?!���Д��?:	��[1��?��[1��?!��[1��?B      ��!       J	�����?�����?!�����?R      ��!       Z	�����?�����?!�����?b      ��!       JGPUY�WÜ��@b q���'@y�b�Á]U@