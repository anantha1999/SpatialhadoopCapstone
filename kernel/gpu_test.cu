//#include "bits/stdc++.h"
//#include<iostream>
//#include<string>
#include<stdio.h>
#include<stdlib.h>

extern "C"

//typedef struct testStruct{
//    int x;
//}testStruct;

__global__ void gpu(float* input, float* output, int* startPoints, int* endPoints, float* distancePoints){
	int block = blockIdx.x;
	output[block*3] = startPoints[block];
	output[block*3+1] = startPoints[block]+1;
	output[block*3+2] = distancePoints[block];

    for(int i1=startPoints[block];i1<endPoints[block];++i1){
        for(int i2=i1+1; i2<endPoints[block];++i2){
            float x_distance = powf(((float)input[i1*2] - (float)input[i2*2]),(float)2);
            float y_distance = powf(((float)input[(i1*2)+1] - (float)input[(i2*2)+1]),(float)2);
            float distance = sqrt(x_distance+y_distance);
            if(distance < output[block*3+2]){
                output[block*3] = (float)i1;
                output[block*3+1] = (float)i2;
                output[block*3+2] = distance;
                }
            }
        }
    }

