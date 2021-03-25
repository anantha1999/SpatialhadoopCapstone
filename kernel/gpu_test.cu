//#include "bits/stdc++.h"
//#include<iostream>
//#include<string>
#include<stdio.h>
#include<stdlib.h>

extern "C"

//typedef struct testStruct{
//    int x;
//}testStruct;

__global__ void gpu(float** input, float* output, int blocks, int size, int lock){
	int i1 = blockIdx.x;
	int i2 = blockIdx.y;
//    printf("Size passed - %d\n",size);
	if(i1 < i2){
//	    printf("Executing %d %d\n",i1,i2);
	    while(i1 < size && i2 < size){
//            printf("Inside while loop! %d %d %d %d\n", input[0][0], input[0][1], input[1][0],input[1][1]);
            float x_distance = powf(((float)input[i1][0] - (float)input[i2][0]),(float)2);
//            printf("x_distance : %f\n",x_distance);
            float y_distance = powf(((float)input[i1][1] - (float)input[i2][1]),(float)2);
	        float distance = sqrt(x_distance+y_distance);
//	        while(lock){printf("%d and %d waiting to enter",i1,i2);}
//	        lock = 1;
//	        printf("%d and %d has locked",i1,i2);
//	        *lock = 0;
	        if(distance < output[2]){
	            output[0] = (float)i1;
	            output[1] = (float)i2;
	            output[2] = distance;
	            lock = 0;
	        }
	        i1 += blocks;
	        i2 += blocks;
	    }
	}
}

