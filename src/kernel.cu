#include <iostream>
#include <vector>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "gpu_structs.hpp"

// basic C string manipulation functions redefinition
__device__ int my_strlen(const char* str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

__device__ static int my_strcmp(const char* str1, const char* str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(unsigned char*)str1 - *(unsigned char*)str2;
}

__device__ static void my_strcpy(char* dest, const char* src) {
    size_t idx = 0;
    while (src[idx] != '\0') {
        dest[idx] = src[idx];
        idx++;
    }
    dest[idx] = '\0';
}

__device__ int my_strncmp(const char* str1, const char* str2, int n) {
    for (int i = 0; i < n; ++i) {
        if (str1[i] == '\0' || str2[i] == '\0') {
            return str1[i] - str2[i];
        }
        if (str1[i] != str2[i]) {
            return str1[i] - str2[i];
        }
    }
    return 0;
}

__global__ static void computeStepKernel(GPUPacket* d_gpuPackets, int numPackets, GPURule* d_gpuRules, int numRules,
    int numCatalysts, int* d_catalystsFlags, int* d_packetsFlags, char** d_gpuCatalystsString, char** d_gpuCatalystsMembraneID) {

    // Assign packet to thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPackets) return;

    // Get thread packet
    GPUPacket& packet = d_gpuPackets[idx];
    if (d_packetsFlags[idx] == 0) return;

    // Iterate on all rules
    for (int i = 0; i < numRules; ++i) {
        GPURule& rule = d_gpuRules[i];

        // If rule and packet belong to the same membrane
        if (my_strcmp(rule.membraneID, packet.membraneID) == 0) {
            bool ruleApplied = false;

            // Iterate on all labels of the packet
            for (int j = 0; j < MAX_SYMBOLS; ++j) {
                if (packet.strings[j][0] == '\0') break;

                // If first condition is true
                if (my_strcmp(packet.strings[j], rule.cond1) == 0) {
                    if (rule.cond2 != nullptr) {
                        bool cond2Found = false;

                        // Look for the second condition in the packet's labels
                        for (int k = 0; k < MAX_SYMBOLS; ++k) {
                            if (packet.strings[k][0] == '\0') break;

                            if (my_strcmp(packet.strings[k], rule.cond2) == 0) {
                                cond2Found = true;
                                // Apply the rule
                                my_strcpy(packet.strings[j], rule.result1);
                                my_strcpy(packet.strings[k], rule.result2);
                                my_strcpy(packet.membraneID, rule.destination);
                                ruleApplied = true;
                                break;
                            }
                        }

#if __CUDA_ARCH__ >= 200
                        // If second condition is not found, it must be a catalyst
                        if (!cond2Found) {
                            for (int c = 0; c < numCatalysts; ++c) {
                                // Ensure exclusive access to the current catalyst
                                if (atomicCAS(&d_catalystsFlags[c], 0, 1) == 0) {

                                    if (my_strcmp(d_gpuCatalystsString[c], rule.cond2) == 0 &&
                                        my_strcmp(d_gpuCatalystsMembraneID[c], packet.membraneID) == 0) {

                                    // Calculate the length of the catalyst string
                                    int catalystLength = my_strlen(d_gpuCatalystsString[c]);

                                    // Check that the length is greater than 2 and compare up to `catalystLength - 2` characters
                                    if (catalystLength > 2 &&
                                        my_strncmp(d_gpuCatalystsString[c], rule.cond2, catalystLength - 2) == 0 &&
                                        my_strcmp(d_gpuCatalystsMembraneID[c], packet.membraneID) == 0) {

                                        // Iterate over all strings in the packet
                                        for (int m = 0; m < MAX_SYMBOLS; ++m) {
                                            if (packet.strings[m][0] == '\0') break;

                                            // Compare packet string with catalyst string (excluding last two chars)
                                            if (my_strncmp(packet.strings[m], d_gpuCatalystsString[c], catalystLength - 2) == 0) {
                                                // Apply the rule
                                                my_strcpy(packet.strings[j], rule.result1);
                                                my_strcpy(packet.membraneID, rule.destination);
                                                my_strcpy(d_gpuCatalystsMembraneID[c], rule.destination);
                                                ruleApplied = true;
                                                break;
                                            }
                                        }
                                     }

                                    }

                                    // Release the lock on the catalyst
                                    atomicExch(&d_catalystsFlags[c], 0);
                                    if (ruleApplied) break; // Exit the loop if the rule was applied
                                }
                               }
                             }
#endif
                    }
                    else {
                        // If second condition doesn't exist, apply the rule directly
                        my_strcpy(packet.strings[j], rule.result1);
                        my_strcpy(packet.membraneID, rule.destination);
                        ruleApplied = true;
                    }
                }
                if (ruleApplied) break;
            }
            if (ruleApplied) break;
        }
    }
}

void computeStepGPU(GPUPacket* d_gpuPackets, int numPackets, GPURule* d_gpuRules, int numRules, int numCatalysts,
    int* d_catalystsFlags, int* d_packetsFlags, char** d_gpuCatalystsString, char** d_gpuCatalystsMembraneID) {

    int threadsPerBlock = 1024; // threads in a block
    int numBlocks = (numPackets + threadsPerBlock - 1) / threadsPerBlock; // blocks

    computeStepKernel << <numBlocks, threadsPerBlock >> > (d_gpuPackets, numPackets, d_gpuRules, numRules, numCatalysts,
        d_catalystsFlags, d_packetsFlags, d_gpuCatalystsString, d_gpuCatalystsMembraneID);

    cudaDeviceSynchronize();

}

// demo (blacklist is not considered)
/* __global__ static void swapCatalystsGPUKernel(int numCatalysts, char** catalystsString, char** catalystsMembraneID) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numCatalysts) return;

    char* string = catalystsString[idx];
    char* membraneID = catalystsMembraneID[idx];

    int s_len = my_strlen(string);
    int m_len = my_strlen(membraneID);

    if (s_len >= 2 && m_len >= 1 && my_strcmp(&string[s_len - 2], "sc") == 0 && membraneID[m_len - 1] == 'd') {
        membraneID[m_len - 1] = 's';
    }

    else if (s_len >= 2 && m_len >= 1 && my_strcmp(&string[s_len - 2], "dc") == 0 && membraneID[m_len - 1] == 's') {
        membraneID[m_len - 1] = 'd';
    }

}

void swapCatalystsGPU(int numCatalysts, char** catalystsString, char** catalystsMembraneID) {

    int threadsPerBlock = 1024; // threads in a block
    int numBlocks = (numCatalysts + threadsPerBlock - 1) / threadsPerBlock; // blocks

    cudaDeviceSynchronize();

    swapCatalystsGPUKernel << <numBlocks, threadsPerBlock >> > (numCatalysts, catalystsString, catalystsMembraneID);

    cudaDeviceSynchronize();
    
} */