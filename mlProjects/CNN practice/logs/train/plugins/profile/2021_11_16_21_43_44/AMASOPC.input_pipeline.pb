	DkE��!@DkE��!@!DkE��!@	��)��@��)��@!��)��@"w
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
	mU��?mU��?!mU��?      ��!       "	��^~��@��^~��@!��^~��@*      ��!       2	����Li�?����Li�?!����Li�?:	���ڧ��?���ڧ��?!���ڧ��?B      ��!       J	�r����?�r����?!�r����?R      ��!       Z	�r����?�r����?!�r����?b      ��!       JGPUY��)��@b q��9j�3@y�l�4�zS@