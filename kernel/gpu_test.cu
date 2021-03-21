//#include "bits/stdc++.h"
//#include<iostream>
//#include<string>
#include<stdio.h>
#include<stdlib.h>
using namespace std;

extern "C"

//class testClass{
//	public:
//		int value = 1100;
//};


__global__ void gpu(int* outputValue){
	printf("Block id : %d Output value before : %d\n", blockIdx.x,*outputValue);
//	testClass obj;
//	printf("Printing class object message %s\n", obj.message);
	outputValue[0] = 1000;
	printf("Output value after : %d",outputValue[0]);
}
