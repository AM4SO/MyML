	���q�$@���q�$@!���q�$@	�3��?�3��?!�3��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���q�$@fi��r�?1衶� @A�V&�R?�?I���3.�?Y�r�]���?*	������T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;ŪA��?!�=J�.@@)p��^�?1N��::@:Preprocessing2U
Iterator::Model::ParallelMapV2Q�\�mO�?!LX� 3@)Q�\�mO�?1LX� 3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�d�?!�RD-�5=@)��Y�N�?1 ���n0@:Preprocessing2F
Iterator::ModelUka�9�?!�+��"A@)	4�ԉ?1����J.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceiV�y˅?!dO�`�)@)iV�y˅?1dO�`�)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorb�� ��t?!�׭k�@)b�� ��t?1�׭k�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipb�G�?!@��nP@)P��W\l?1��@��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 11.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�3��?I0	n�}J/@Qrp�H־T@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	fi��r�?fi��r�?!fi��r�?      ��!       "	衶� @衶� @!衶� @*      ��!       2	�V&�R?�?�V&�R?�?!�V&�R?�?:	���3.�?���3.�?!���3.�?B      ��!       J	�r�]���?�r�]���?!�r�]���?R      ��!       Z	�r�]���?�r�]���?!�r�]���?b      ��!       JGPUY�3��?b q0	n�}J/@yrp�H־T@