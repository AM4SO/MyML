	z�0��/@z�0��/@!z�0��/@	hC��Z�?hC��Z�?!hC��Z�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6z�0��/@���N@��?1? �M��+@A/��C?�?I���"1A�?YI���p��?*	V-��W@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����qߚ?!iNj��a;@)����?��?1�&��i7@:Preprocessing2U
Iterator::Model::ParallelMapV2���)�?!�u�=~}5@)���)�?1�u�=~}5@:Preprocessing2F
Iterator::Model�I��{�?!���D@)U��7��?1��c��@4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��u�B�?!@��/�9@)e����c�?1��}�m�.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceMHk:!�?!�]g���$@)MHk:!�?1�]g���$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<�$��?!rs��� M@)��~P)t?1��'��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���)o?!1?�-��@)���)o?11?�-��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 8.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9hC��Z�?I؜%0�.&@QW�B��U@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���N@��?���N@��?!���N@��?      ��!       "	? �M��+@? �M��+@!? �M��+@*      ��!       2	/��C?�?/��C?�?!/��C?�?:	���"1A�?���"1A�?!���"1A�?B      ��!       J	I���p��?I���p��?!I���p��?R      ��!       Z	I���p��?I���p��?!I���p��?b      ��!       JGPUYhC��Z�?b q؜%0�.&@yW�B��U@