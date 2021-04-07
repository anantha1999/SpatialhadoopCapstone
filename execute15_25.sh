nvcc -ptx kernel/gpu_test.cu -o gpu_test.ptx
hdfs dfs -put -f gpu_test.ptx .
hadoop jar target/spatialhadoop-2.4.1-SNAPSHOT-uber.jar closestpair test15 output15gbCPU shape:point -overwrite
hadoop jar target/spatialhadoop-2.4.1-SNAPSHOT-uber.jar closestpair 25gb output25gbCPU shape:point -overwrite
