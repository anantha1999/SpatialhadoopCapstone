nvcc -ptx kernel/gpu_test.cu -o gpu_test.ptx
hdfs dfs -put -f gpu_test.ptx .
hadoop jar target/spatialhadoop-2.4.1-SNAPSHOT-uber.jar closestpair 360mb output360mbGPU shape:point -overwrite
