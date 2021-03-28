//#include "bits/stdc++.h"
//#include<iostream>
//#include<string>
#include<stdio.h>
#include<stdlib.h>

extern "C"

//typedef struct testStruct{
//    int x;
//}testStruct;

__global__ void gpu(float** input, float** output, int blocks, int size, int *lock){
	int block = blockIdx.x;
	int thread = threadIdx.x;
	int i1 = block;
	int i2 = thread;
//    printf("Size passed - %d Lock - %d blocks - %d\n",size, lock, blocks);
//    printf("Inputs - %d %d Outputs - %d %d %d\n", input[i1][0], input[i1][1], output[0], output[1], output[2]);
	if(i1 < i2){
	    while(i1 < size && i2 < size){
            float x_distance = powf(((float)input[i1][0] - (float)input[i2][0]),(float)2);
            float y_distance = powf(((float)input[i1][1] - (float)input[i2][1]),(float)2);
	        float distance = sqrt(x_distance+y_distance);


	        if(distance < output[block][2]){
	            output[block][0] = (float)i1;
	            output[block][1] = (float)i2;
	            output[block][2] = distance;
	        }
	        i1 += blocks;
	        i2 += blocks;
	    }
	}
}

