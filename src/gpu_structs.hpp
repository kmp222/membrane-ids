#ifndef GPU_STRUCTS_HPP
#define GPU_STRUCTS_HPP

#include <string>
#include <vector>

#define MAX_SYMBOLS 9
#define MAX_SYMBOL_LENGTH 5

struct GPUPacket {

	char strings[MAX_SYMBOLS][MAX_SYMBOL_LENGTH];
	char* membraneID;

};

struct GPUCatalyst {
	
	char* string;
	char* membraneID;

};

struct GPURule {

	char* cond1;
	char* cond2;
	char* result1;
	char* result2;
	char* destination;
	char* membraneID;

};

#endif