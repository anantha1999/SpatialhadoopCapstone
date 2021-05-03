nvcc -ptx kernel/gpu_test.cu -o gpu_test.ptx
hdfs dfs -put -f gpu_test.ptx .
