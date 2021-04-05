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
//	int i1 = blockIdx.y + startPoints[block];
//	int i2 = blockIdx.z + startPoints[block];
//	if(i2 > i1){
//	    if(i1 < endPoints[block] && i2 < endPoints[block]){
//	        float x_distance = powf(((float)input[i1][0] - (float)input[i2][0]),(float)2);
//            float y_distance = powf(((float)input[i1][1] - (float)input[i2][1]),(float)2);
//            float distance = sqrt(x_distance+y_distance);
//           if(distance < output[block][2]){
//                output[block][0] = (float)i1;
//                output[block][1] = (float)i2;
//                output[block][2] = distance;
//            }
//	    }
//	}
    for(int i1=startPoints[block];i1<endPoints[block];++i1){
        for(int i2=i1+1; i2<endPoints[block];++i2){
            // printf("Input points - %f %f %f %f\n",input[i1*2],input[i1*2+1],input[i2*2],input[i2*2+1]);
            float x_distance = powf(((float)input[i1*2] - (float)input[i2*2]),(float)2);
            float y_distance = powf(((float)input[(i1*2)+1] - (float)input[(i2*2)+1]),(float)2);
            float distance = sqrt(x_distance+y_distance);
//            printf("distance calculated - %d\n",distance);
            if(distance < output[block*3+2]){
                // printf("Input points - %f %f %f %f\n",input[i1*2],input[(i1*2)+1],input[i2*2],input[(i2*2)+1]);
                output[block*3] = (float)i1;
                output[block*3+1] = (float)i2;
                output[block*3+2] = distance;
                }
            }
        }
    }

