	H�}8[4@H�}8[4@!H�}8[4@	.*���U�?.*���U�?!.*���U�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6H�}8[4@�8�ߡ��?1��uR_:2@A����?IS��%���?Yg�����?*	����xma@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat6 B\9{�?!��s��r@@)�5w��\�?1�N��;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�v�ӂ�?!�m�7�=@)!�> �M�?1��-)�3@:Preprocessing2F
Iterator::Modelvi���?!�ӵ���A@)�eo)�?1�6_�]�3@:Preprocessing2U
Iterator::Model::ParallelMapV2B"m�OT�?!<�P�G/@)B"m�OT�?1<�P�G/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��M�?!�U��q#@)��M�?1�U��q#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorҩ+��y�?!�da��@)ҩ+��y�?1�da��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$���9"�?!�<,4P@)��	�yk?1��v]��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 7.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9-*���U�?I���a>!@Qut{��bV@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�8�ߡ��?�8�ߡ��?!�8�ߡ��?      ��!       "	��uR_:2@��uR_:2@!��uR_:2@*      ��!       2	����?����?!����?:	S��%���?S��%���?!S��%���?B      ��!       J	g�����?g�����?!g�����?R      ��!       Z	g�����?g�����?!g�����?b      ��!       JGPUY-*���U�?b q���a>!@yut{��bV@