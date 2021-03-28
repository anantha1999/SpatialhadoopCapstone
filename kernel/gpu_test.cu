//#include "bits/stdc++.h"
//#include<iostream>
//#include<string>
#include<stdio.h>
#include<stdlib.h>

extern "C"

//typedef struct testStruct{
//    int x;
//}testStruct;

__global__ void gpu(float** input, float** output, int* startPoints, int* endPoints){
	int block = blockIdx.x;
    for(int i1=startPoints[block];i1<endPoints[block];++i1){
        for(int i2=i1+1; i2<endPoints[block];++i2){
            float x_distance = powf(((float)input[i1][0] - (float)input[i2][0]),(float)2);
            float y_distance = powf(((float)input[i1][1] - (float)input[i2][1]),(float)2);
            float distance = sqrt(x_distance+y_distance);
//            printf("distance calculated - %d\n",distance);
            if(distance < output[block][2]){
                output[block][0] = (float)i1;
                output[block][1] = (float)i2;
                output[block][2] = distance;
            }
        }
    }
}

