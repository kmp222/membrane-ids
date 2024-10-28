#include <iostream>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <iomanip>
#include <map>
#include <random>

#include <chrono>
#include <thread>
#include <time.h>

#include <cuda_runtime.h>

#include "cell.hpp"
#include "gpu_structs.hpp"

std::vector<std::pair<std::string, std::string>> blacklist;

// applies one-step rules to packets (gpu kernel)
extern void computeStepGPU(GPUPacket* d_gpuPackets, int numPackets, GPURule* d_gpuRules, int numRules, int numCatalysts,
	int* d_catalystsFlags, int* d_packetsFlags, char** d_gpuCatalystsString, char** d_gpuCatalystsMembraneID);

// demo (blacklist is not considered)
// 
// extern void swapCatalystsGPU(int numCatalysts, char** catalystsString, char** catalystsMembraneID);

// applies one-step rules to packets (cpu)
static void computeStepCPU(GPUPacket* d_gpuPackets, int numPackets, GPURule* d_gpuRules, int numRules, int numCatalysts,
	int* d_catalystsFlags, int* d_packetsFlags, char** d_gpuCatalystsString, char** d_gpuCatalystsMembraneID) {

	for (int p = 0; p < numPackets; ++p) {

		GPUPacket& packet = d_gpuPackets[p];
		if (d_packetsFlags[p] == 0) continue;

		for (int i = 0; i < numRules; ++i) {

			GPURule& rule = d_gpuRules[i];

			if (strcmp(rule.membraneID, packet.membraneID) == 0) {
				bool ruleApplied = false;

				for (int j = 0; j < MAX_SYMBOLS; ++j) {
					if (packet.strings[j][0] == '\0') break;

					if (strcmp(packet.strings[j], rule.cond1) == 0) {
						if (rule.cond2 != nullptr) {
							bool cond2Found = false;

							for (int k = 0; k < MAX_SYMBOLS; ++k) {
								if (packet.strings[k][0] == '\0') break;

								if (strcmp(packet.strings[k], rule.cond2) == 0) {
									cond2Found = true;
									strcpy(packet.strings[j], rule.result1);
									strcpy(packet.strings[k], rule.result2);
									strcpy(packet.membraneID, rule.destination);
									ruleApplied = true;
									break;
								}
							}

							if (!cond2Found) {
								for (int c = 0; c < numCatalysts; ++c) {
									if (strcmp(d_gpuCatalystsString[c], rule.cond2) == 0 &&
										strcmp(d_gpuCatalystsMembraneID[c], packet.membraneID) == 0) {

										for (int m = 0; m < MAX_SYMBOLS; ++m) {
											if (packet.strings[m][0] == '\0') break;

											int catalystLength = strlen(d_gpuCatalystsString[c]);

											if (catalystLength > 2 &&
												strncmp(packet.strings[m], d_gpuCatalystsString[c], catalystLength - 2) == 0) {

												strcpy(packet.strings[j], rule.result1);
												strcpy(packet.membraneID, rule.destination);
												strcpy(d_gpuCatalystsMembraneID[c], rule.destination);
												ruleApplied = true;
												break;
											}
										}

										if (ruleApplied) break;
									}
								}
							}

						}
						else {
							strcpy(packet.strings[j], rule.result1);
							strcpy(packet.membraneID, rule.destination);
							ruleApplied = true;
						}
					}
					if (ruleApplied) break;
				}
				if (ruleApplied) break;
			}

		}
	}

}

// adapts catalysts and packets for gpu computation
static void convertObjects(std::vector<GPUPacket>& gpuPackets, std::vector<GPUCatalyst>& gpuCatalysts, Membrane* membrane) {
	for (const auto& obj : membrane->objects) {
		if (obj.size() == 1) {

			GPUCatalyst gpuCatalyst;

			gpuCatalyst.string = new char[obj[0].size() + 1];
			std::strcpy(gpuCatalyst.string, obj[0].c_str());

			gpuCatalyst.membraneID = new char[membrane->ID.size() + 1];
			std::strcpy(gpuCatalyst.membraneID, membrane->ID.c_str());

			gpuCatalysts.push_back(gpuCatalyst);
		}
		else {

			GPUPacket gpuPacket;

			size_t numStrings = std::min(obj.size(), static_cast<size_t>(MAX_SYMBOLS));
			for (size_t i = 0; i < numStrings; ++i) {
				std::strncpy(gpuPacket.strings[i], obj[i].c_str(), MAX_SYMBOL_LENGTH);
				gpuPacket.strings[i][MAX_SYMBOL_LENGTH - 1] = '\0';
			}

			for (size_t i = numStrings; i < MAX_SYMBOLS; ++i) {
				gpuPacket.strings[i][0] = '\0';
			}

			gpuPacket.membraneID = new char[membrane->ID.size() + 1];
			std::strcpy(gpuPacket.membraneID, membrane->ID.c_str());

			gpuPackets.push_back(gpuPacket);
		}
	}
}

// rules are adapted for gpu computation
static void convertRules(std::vector<GPURule>& gpuRules, Membrane* membrane) {
	for (const auto& rule : membrane->rules) {
		GPURule gpuRule;
		gpuRule.cond1 = (rule.conditions.size() > 0) ? new char[rule.conditions[0].size() + 1] : nullptr;
		if (gpuRule.cond1) std::strcpy(gpuRule.cond1, rule.conditions[0].c_str());

		gpuRule.cond2 = (rule.conditions.size() > 1) ? new char[rule.conditions[1].size() + 1] : nullptr;
		if (gpuRule.cond2) std::strcpy(gpuRule.cond2, rule.conditions[1].c_str());

		gpuRule.result1 = (rule.result.size() > 0) ? new char[rule.result[0].size() + 1] : nullptr;
		if (gpuRule.result1) std::strcpy(gpuRule.result1, rule.result[0].c_str());

		gpuRule.result2 = (rule.result.size() > 1) ? new char[rule.result[1].size() + 1] : nullptr;
		if (gpuRule.result2) std::strcpy(gpuRule.result2, rule.result[1].c_str());

		gpuRule.destination = new char[rule.destination.size() + 1];
		std::strcpy(gpuRule.destination, rule.destination.c_str());

		gpuRule.membraneID = new char[membrane->ID.size() + 1];
		std::strcpy(gpuRule.membraneID, membrane->ID.c_str());

		gpuRules.push_back(gpuRule);
	}
}

// activates a % of packets randomly chosen among those inactive
static void activatePackets(int* d_packetsFlags, int numPackets, int percentage) {

	int totalPacketsToActivate = 0;

	for (int i = 0; i < numPackets; ++i) {
		if (d_packetsFlags[i] == 0) {
			totalPacketsToActivate++;
		}
	}

	float packetsToActivate = (totalPacketsToActivate * percentage) / 100.0f;

	if (packetsToActivate < 1 && packetsToActivate > 0) {
		packetsToActivate = 1;
	}

	int packetsToActivateInt = static_cast<int>(packetsToActivate);

	if (packetsToActivateInt == 0) {
		return;
	}

	std::vector<int> indices;

	for (int i = 0; i < numPackets; ++i) {
		if (d_packetsFlags[i] == 0) {
			indices.push_back(i);
		}
	}

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(indices.begin(), indices.end(), g);

	for (int i = 0; i < packetsToActivateInt; ++i) {
		d_packetsFlags[indices[i]] = 1;
	}

}

// activates a random packet in a nucleus
static void activatePacketInNucleus(GPUPacket* d_gpuPackets, int numPackets, int* d_packetsFlags, const char* nucleusID) {
	std::vector<int> indices;

	for (int i = 0; i < numPackets; ++i) {
		if (strcmp(d_gpuPackets[i].membraneID, nucleusID) == 0 && d_packetsFlags[i] == 0) {
			indices.push_back(i);
		}
	}

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(indices.begin(), indices.end(), g);

	for (int i = 0; i < indices.size(); ++i) {
		d_packetsFlags[indices[i]] = 1;
	}

}

// reads a tree topology (network) from file and convert it into membranes
static void treeFileToCell(const std::string& filename, Cell& cell) {

	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "unable to open cell input file" << std::endl;
		return;
	}

	std::string line;
	std::unordered_set<std::string> routers;
	std::unordered_set<std::string> machines;

	std::getline(file, line);
	std::string rootLabel = line;
	routers.insert(rootLabel);

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string parent, child;
		if (std::getline(ss, parent, '-') && std::getline(ss, child, ',')) {
			routers.insert(parent);
			if (routers.find(child) == routers.end()) {
				machines.insert(child);
			}
		}
	}

	file.clear();
	file.seekg(0, std::ios::beg);
	std::getline(file, line);

	Membrane* rootMembrane = new Membrane(rootLabel, MembraneType::Router);
	cell.rootMembrane = rootMembrane;
	cell.addMembrane("", rootLabel, MembraneType::Router);

	cell.addMembrane(rootLabel, rootLabel + "s", MembraneType::SrcCheck);
	cell.addMembrane(rootLabel + "s", rootLabel + "d", MembraneType::DstCheck);

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string parent, child;
		if (std::getline(ss, parent, '-') && std::getline(ss, child, ',')) {
			MembraneType parentType = MembraneType::Router;
			MembraneType childType = routers.find(child) != routers.end() ? MembraneType::Router : MembraneType::Machine;

			cell.addMembrane(parent, child, childType);

			if (childType == MembraneType::Router) {
				cell.addMembrane(child, child + "s", MembraneType::SrcCheck);
				cell.addMembrane(child + "s", child + "d", MembraneType::DstCheck);
			}
			else if (childType == MembraneType::Machine) {
				cell.addMembrane(child, child + "n", MembraneType::Nucleus);
			}
		}
	}

	file.close();
}

// reads packets from file to place them into machine membranes at the beginning of the simulation
static void fileToPackets(const std::string& filename, Cell& cell) {

	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "unable to open packets input file" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(file, line)) {

		std::stringstream ss(line);
		std::string path;
		int count;

		std::getline(ss, path, ' ');
		ss >> count;

		std::vector<std::string> obj;
		std::stringstream pathStream(path);
		std::string segment;

		while (std::getline(pathStream, segment, '-')) {
			obj.push_back(segment);
		}

		for (int i = 0; i < count; ++i) {
			cell.findMembrane(obj.front() + std::string("n"))->objects.push_back(obj);
		}

	}

}

// reads catalysts from file to insert them in the qty specificed
// into srcChecks and dstChecks membranes at the beginning of the simulation
//
// note: if some catalyst quantity is not specified, 'infinite' is assumed (no alerts will ever be generated).
static void fileToCatalysts(const std::string& filename, Cell& cell) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "unable to open catalysts input file" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string catalystID;
		int quantity;

		std::getline(ss, catalystID, ' ');
		ss >> quantity;

		std::string suffix = catalystID.substr(catalystID.size() - 2);
		std::vector<Membrane*> targetMembranes;

		std::function<void(Membrane*)> findTargetMembranes = [&](Membrane* membrane) {
			if (membrane->ID.size() > 1 && membrane->ID.substr(membrane->ID.size() - 1) == suffix.substr(0, 1)) {
				targetMembranes.push_back(membrane);
			}
			for (auto& subMembrane : membrane->subMembranes) {
				findTargetMembranes(subMembrane);
			}
		};

		findTargetMembranes(cell.rootMembrane);

		for (Membrane* membrane : targetMembranes) {
			for (int j = 0; j < quantity; ++j) {
				membrane->objects.push_back({ catalystID });
			}
		}
	}

	file.close();
}

// reads catalysts from file to insert them in the qty specificed
// into specific srcChecks and dstChecks membranes at the beginning of the simulation
static void fileToCustomCatalysts(const std::string& filename, Cell& cell) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "unable to open custom catalysts input file" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string catalystID;
		int quantity;
		int routerLabel;

		std::getline(ss, catalystID, ' ');
		ss >> quantity >> routerLabel;

		std::string suffix;
		if (catalystID.size() > 2 && catalystID.substr(catalystID.size() - 2) == "sc") {
			suffix = std::to_string(routerLabel) + "s";
		}
		else if (catalystID.size() > 2 && catalystID.substr(catalystID.size() - 2) == "dc") {
			suffix = std::to_string(routerLabel) + "d";
		}
		else {
			std::cerr << "invalid catalyst type for catalyst: " << catalystID << std::endl;
			continue;
		}

		std::vector<Membrane*> targetMembranes;

		std::function<void(Membrane*)> findTargetMembranes = [&](Membrane* membrane) {
			if (membrane->ID == suffix) {
				targetMembranes.push_back(membrane);
			}
			for (auto& subMembrane : membrane->subMembranes) {
				findTargetMembranes(subMembrane);
			}
			};

		findTargetMembranes(cell.rootMembrane);

		for (Membrane* membrane : targetMembranes) {
			for (int j = 0; j < quantity; ++j) {
				membrane->objects.push_back({ catalystID });
			}
		}
	}

	file.close();

}

// saves the initial catalysts configuration in every membrane to use for resource check
static void storeCatalystsInfo(Cell& cell, std::map<std::string, std::map<std::string, int>>& routerCatalystsQty) {

	std::function<void(Membrane*)> func = [&](Membrane* membrane) {
		std::string id = membrane->ID;
		std::string suffix;

		if (id.back() == 's') {
			suffix = "sc";
		}
		else if (id.back() == 'd') {
			suffix = "dc";
		}

		if (!suffix.empty()) {
			if (routerCatalystsQty.find(id) == routerCatalystsQty.end()) {
				routerCatalystsQty[id] = std::map<std::string, int>();
			}

			for (const auto& obj : membrane->objects) {
				if (obj.size() == 1 && obj[0].find(suffix) != std::string::npos) {
					std::string catalyst = obj[0];
					routerCatalystsQty[id][catalyst]++;
				}
			}
		}

		for (auto& child : membrane->subMembranes) {
			func(child);
		}
		};

	func(cell.rootMembrane);

}

// checks catalysts quantities globally to trigger alerts and save info on log file
static std::string checkAlerts(int threshold, char** membraneIDs, char** catalysts, int numCatalysts,
	const std::map<std::string, std::map<std::string, int>>& routerCatalystsQty, int iteration) {

	std::stringstream resultMessage;

	std::map<std::string, std::map<std::string, int>> currentCatalystsQty;

	for (int i = 0; i < numCatalysts; ++i) {
		std::string membraneID = membraneIDs[i];
		std::string catalyst = catalysts[i];

		currentCatalystsQty[membraneID][catalyst]++;
	}

	for (const auto& entry : routerCatalystsQty) {
		const std::string& membraneID = entry.first;
		const std::map<std::string, int>& initialCatalysts = entry.second;

		for (const auto& catalystEntry : initialCatalysts) {
			const std::string& catalyst = catalystEntry.first;
			int initialCount = catalystEntry.second;
			int currentCount = currentCatalystsQty[membraneID][catalyst];

			if (initialCount > 0) {
				float percentageRemaining = (currentCount / static_cast<float>(initialCount)) * 100;

				std::string anomalousMachine = catalyst.substr(0, catalyst.size() - 2);
				std::string action = (catalyst.substr(catalyst.size() - 2) == "sc") ?
					(percentageRemaining == 0 ? "CAN NOT SEND" : "SENDING") :
					(percentageRemaining == 0 ? "CAN NOT RECEIVE" : "RECEIVING");
				std::string blockingRouter = membraneID.substr(0, membraneID.size() - 1);

				std::string message;

				if (static_cast<int>(percentageRemaining) == 0) {
					
					message = "BLOCKED: " + anomalousMachine + " " + action +
						" ANYMORE VIA " + blockingRouter; 

					resultMessage << "iteration " << iteration << ": " << message << "\n";

					if (action == "CAN NOT SEND") {

						if (std::find(blacklist.begin(), blacklist.end(),
							std::make_pair(anomalousMachine, blockingRouter)) == blacklist.end()) {
							blacklist.push_back(std::make_pair(anomalousMachine, blockingRouter));
						}

					}

				}
				else if (percentageRemaining <= threshold) {
					
					message = anomalousMachine + " IS " + action + " TOO MANY PACKETS VIA " +
						blockingRouter + " (" + std::to_string(static_cast<int>(percentageRemaining)) + "% resources left)"; 

					resultMessage << "iteration " << iteration << ": " << message << "\n";

					if (action == "SENDING") {

						if (std::find(blacklist.begin(), blacklist.end(),
							std::make_pair(anomalousMachine, blockingRouter)) == blacklist.end()) {
							blacklist.push_back(std::make_pair(anomalousMachine, blockingRouter));
						}

					}

				}
			}
		}
	}

	return resultMessage.str();

}

// resets all catalysts to their original configuration, blacklisted senders excluded
static void swapCatalystsCPU(int numCatalysts, char** d_gpuCatalystsString, char** d_gpuCatalystsMembraneID) {

	for (int j = 0; j < numCatalysts; ++j) {

		int s_len = strlen(d_gpuCatalystsString[j]);
		int m_len = strlen(d_gpuCatalystsMembraneID[j]);

		if (strcmp(&d_gpuCatalystsString[j][s_len - 2], "sc") == 0 &&
			d_gpuCatalystsMembraneID[j][m_len - 1] == 'd') {

			std::string machineLabel(d_gpuCatalystsString[j], s_len - 2);

			std::string blockingRouter(d_gpuCatalystsMembraneID[j], m_len - 1);

			std::pair<std::string, std::string> keyPair = std::make_pair(machineLabel, blockingRouter);

			if (std::find(blacklist.begin(), blacklist.end(), keyPair) != blacklist.end()) {
				continue;
			}

		}

		if (strcmp(&d_gpuCatalystsString[j][s_len - 2], "sc") == 0 &&
			d_gpuCatalystsMembraneID[j][m_len - 1] == 'd') {
			d_gpuCatalystsMembraneID[j][m_len - 1] = 's';

		}

		else if (strcmp(&d_gpuCatalystsString[j][s_len - 2], "dc") == 0 &&
			d_gpuCatalystsMembraneID[j][m_len - 1] == 's') {
			d_gpuCatalystsMembraneID[j][m_len - 1] = 'd';

		}

	}

}

// prints catalysts in each router
void printRouterCatalystsQty(const std::map<std::string, std::map<std::string, int>>& routerCatalystsQty) {

	std::cout << std::endl;

	for (const auto& router : routerCatalystsQty) {
		std::cout << "Router: " << router.first << std::endl;

		for (const auto& catalyst : router.second) {
			std::cout << "  Catalyst: " << catalyst.first << ", Quantity: " << catalyst.second << std::endl;
		}

		std::cout << std::endl;
	}
}

int main() {

	// init cell
	Cell cell{};

	std::cout << "Loading files ..." << std::endl;

	// read input files
	treeFileToCell("input/tree_32.txt", cell);
	fileToPackets("input/packets_32.txt", cell);
	fileToCatalysts("input/catalysts_32.txt", cell);
	fileToCustomCatalysts("input/catalysts_custom_32.txt", cell);

	std::cout << "Files loaded ..." << std::endl;
	std::cout << "Converting data ..." << std::endl;

	// show content of the cell (membranes and object inside each)
	// cell.print();

	// store catalysts starting configuration
	std::map<std::string, std::map<std::string, int>> routerCatalystsQty;
	storeCatalystsInfo(cell, routerCatalystsQty);

	// printRouterCatalystsQty(routerCatalystsQty);

	cell.generateRules();

	// convert data for gpu computation
	std::vector<GPUPacket> gpuPackets;
	std::vector<GPURule> gpuRules;
	std::vector<GPUCatalyst> gpuCatalysts;

	std::function<void(Membrane*)> convertMembrane = [&](Membrane* membrane) {
		convertObjects(gpuPackets, gpuCatalysts, membrane);
		convertRules(gpuRules, membrane);
		for (Membrane* subMembrane : membrane->subMembranes) {
			convertMembrane(subMembrane);
		}
	};

	convertMembrane(cell.rootMembrane);

	std::cout << "Data converted ..." << std::endl;
	std::cout << "Allocating GPU memory ..." << std::endl;

	// allocate memory for gpu

	// allocate packets and rules on Unified Memory

	std::cout << "... cudaMallocManaged calls ..." << std::endl;

	GPUPacket* d_gpuPackets;
	GPURule* d_gpuRules;
	int* d_catalystsFlags; // handle locks to ensure atomic ops on device
	int* d_packetsFlags; // 0: inactive, 1: active

	size_t numPackets = gpuPackets.size();
	size_t numRules = gpuRules.size();
	size_t numCatalysts = gpuCatalysts.size();


	cudaMallocManaged(&d_gpuPackets, sizeof(GPUPacket) * numPackets);
	cudaMallocManaged(&d_gpuRules, sizeof(GPURule) * numRules);
	cudaMallocManaged(&d_catalystsFlags, sizeof(int) * numCatalysts);
	cudaMallocManaged(&d_packetsFlags, sizeof(int) * numPackets);

	// fill allocated memory
	for (int i = 0; i < numPackets; ++i) {
		GPUPacket& gpuPacket = gpuPackets[i];

		cudaMallocManaged(&d_gpuPackets[i].membraneID, (strlen(gpuPacket.membraneID) + 1) * sizeof(char));
		strcpy(d_gpuPackets[i].membraneID, gpuPacket.membraneID);

		for (int j = 0; j < MAX_SYMBOLS; ++j) {
			if (strlen(gpuPacket.strings[j]) > 0) {
				strncpy(d_gpuPackets[i].strings[j], gpuPacket.strings[j], MAX_SYMBOL_LENGTH);
			}
			else {
				d_gpuPackets[i].strings[j][0] = '\0';
			}
		}
	}

	for (int i = 0; i < numRules; ++i) {

		GPURule& gpuRule = gpuRules[i];

		if (gpuRule.cond1 != nullptr) {
			cudaMallocManaged(&d_gpuRules[i].cond1, (strlen(gpuRule.cond1) + 1) * sizeof(char));
			strcpy(d_gpuRules[i].cond1, gpuRule.cond1);
		}
		else {
			d_gpuRules[i].cond1 = nullptr;
		}

		if (gpuRule.cond2 != nullptr) {
			cudaMallocManaged(&d_gpuRules[i].cond2, (strlen(gpuRule.cond2) + 1) * sizeof(char));
			strcpy(d_gpuRules[i].cond2, gpuRule.cond2);
		}
		else {
			d_gpuRules[i].cond2 = nullptr;
		}

		if (gpuRule.result1 != nullptr) {
			cudaMallocManaged(&d_gpuRules[i].result1, (strlen(gpuRule.result1) + 1) * sizeof(char));
			strcpy(d_gpuRules[i].result1, gpuRule.result1);
		}
		else {
			d_gpuRules[i].result1 = nullptr;
		}

		if (gpuRule.result2 != nullptr) {
			cudaMallocManaged(&d_gpuRules[i].result2, (strlen(gpuRule.result2) + 1) * sizeof(char));
			strcpy(d_gpuRules[i].result2, gpuRule.result2);
		}
		else {
			d_gpuRules[i].result2 = nullptr;
		}

		cudaMallocManaged(&d_gpuRules[i].destination, (strlen(gpuRule.destination) + 1) * sizeof(char));
		strcpy(d_gpuRules[i].destination, gpuRule.destination);

		cudaMallocManaged(&d_gpuRules[i].membraneID, (strlen(gpuRule.membraneID) + 1) * sizeof(char));
		strcpy(d_gpuRules[i].membraneID, gpuRule.membraneID);
	}
	
	// allocate catalysts on Pinned Memory

	std::cout << "... cudaHostAlloc calls ..." << std::endl;

	char** d_gpuCatalystsString;
	char** d_gpuCatalystsMembraneID;

	cudaHostAlloc(&d_gpuCatalystsString, sizeof(char*)* numCatalysts, cudaHostAllocMapped);
	cudaHostAlloc(&d_gpuCatalystsMembraneID, sizeof(char*)* numCatalysts, cudaHostAllocMapped);

	char** h_gpuCatalystsString = (char**)malloc(sizeof(char*) * numCatalysts);
	char** h_gpuCatalystsMembraneID = (char**)malloc(sizeof(char*) * numCatalysts);

	// temp data structures to save catalysts after gpu computation
	char** h_tempCatalystsString = (char**)malloc(sizeof(char*) * numCatalysts);

	for (int i = 0; i < numCatalysts; i++) {
		GPUCatalyst& gpuCatalyst = gpuCatalysts[i];

		h_gpuCatalystsString[i] = (char*)malloc((strlen(gpuCatalyst.string) + 1) * sizeof(char));
		h_tempCatalystsString[i] = (char*)malloc((strlen(gpuCatalyst.string) + 1) * sizeof(char));
		strcpy(h_gpuCatalystsString[i], gpuCatalyst.string);
		strcpy(h_tempCatalystsString[i], gpuCatalyst.string);

		h_gpuCatalystsMembraneID[i] = (char*)malloc((strlen(gpuCatalyst.membraneID) + 1) * sizeof(char));
		strcpy(h_gpuCatalystsMembraneID[i], gpuCatalyst.membraneID);
	}

	for (int i = 0; i < numCatalysts; i++) {
		char* d_string;
		cudaHostAlloc(&d_string, (strlen(h_gpuCatalystsString[i]) + 1) * sizeof(char), cudaHostAllocMapped);
		cudaMemcpy(d_string, h_gpuCatalystsString[i], (strlen(h_gpuCatalystsString[i]) + 1) * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_gpuCatalystsString[i], &d_string, sizeof(char*), cudaMemcpyHostToDevice);

		char* d_membraneID;
		cudaHostAlloc(&d_membraneID, (strlen(h_gpuCatalystsMembraneID[i]) + 1) * sizeof(char), cudaHostAllocMapped);
		cudaMemcpy(d_membraneID, h_gpuCatalystsMembraneID[i], (strlen(h_gpuCatalystsMembraneID[i]) + 1) * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_gpuCatalystsMembraneID[i], &d_membraneID, sizeof(char*), cudaMemcpyHostToDevice);
	}

	cudaMemset(d_catalystsFlags, 0, sizeof(int) * numCatalysts);
	cudaMemset(d_packetsFlags, 0, sizeof(int) * numPackets);

	std::cout << "GPU memory allocated ..." << std::endl;
	std::cout << "Simulator ready!" << std::endl;
	std::cout << std::endl;

	// data structure to save attackers
	// for each router decided by the user, one packet gets activated iteratively
	std::vector <char*> sourceMembranesForIntensiveTraffic;

	// input simulation parameters
	int nIter,
		packetsClock,
		packetsToActivatePercentage,
		initialActivePacketsPercentage,
		catalystsClock,
		thresholdCatalystsAlertPercentage,
		checksClock;

	bool useGPU = false;
	char computeChoice;

	std::cout << "# iterations: ";
	std::cin >> nIter;

	std::cout << "% packets to randomly activate: ";
	std::cin >> initialActivePacketsPercentage;

	std::cout << "# iterations after which some packets are cyclically activated: ";
	std::cin >> packetsClock;

	int packetsClockValue = packetsClock;

	std::cout << "% packets to cyclically and randomly activate after " << packetsClock << " iteration(s): ";
	std::cin >> packetsToActivatePercentage;

	std::cout << "# iterations after which catalysts are cyclically reset: ";
	std::cin >> catalystsClock;

	int catalystsClockValue = catalystsClock;

	std::cout << "% threshold remaining specific catalyst to trigger alerts: ";
	std::cin >> thresholdCatalystsAlertPercentage;

	std::cout << "# iterations after which a check is cyclically done: ";
	std::cin >> checksClock;

	int intensiveTrafficIterations;
	std::cout << "# iterations after which machines can cyclically be selected to generate intensive traffic: ";
	std::cin >> intensiveTrafficIterations;
	int intensiveTrafficIterationsClock = intensiveTrafficIterations;

	std::cout << "compute on GPU (y/n)? ";
	std::cin >> computeChoice;

	if (computeChoice == 'y' || computeChoice == 'Y') {
		useGPU = true;
	}

	std::ofstream logsFile("output/check_logs.txt", std::ios::trunc);
	logsFile.close();
	logsFile.open("output/check_logs.txt", std::ios::app);

	double time = 0;
	double timedif = 0;

	activatePackets(d_packetsFlags, numPackets, initialActivePacketsPercentage);

	// execute simulation
	if (useGPU) {

		std::ofstream rulesFile("output/rules_time_gpu.txt", std::ios::trunc);
		std::ofstream cudaSyncFile("output/cudasync_time_gpu.txt", std::ios::trunc);

		rulesFile.close();
		cudaSyncFile.close();

		rulesFile.open("output/rules_time_gpu.txt", std::ios::app);
		cudaSyncFile.open("output/cudasync_time_gpu.txt", std::ios::app);

		for (int i = 0; i < nIter; ++i) {

			double cudaTime = 0;
			double endCudaTime = 0;

			time = (double)clock();
			time = time / CLOCKS_PER_SEC;

			computeStepGPU(d_gpuPackets, numPackets, d_gpuRules, numRules, numCatalysts,
				d_catalystsFlags, d_packetsFlags, d_gpuCatalystsString, d_gpuCatalystsMembraneID);

			timedif = (((double)clock())) / CLOCKS_PER_SEC - time;

			rulesFile << timedif << std::endl;

			if ((i+1) % checksClock == 0) {

				cudaTime = (double)clock() / CLOCKS_PER_SEC;
				cudaDeviceSynchronize(); // just in case
				endCudaTime = (((double)clock())) / CLOCKS_PER_SEC - cudaTime;

				std::string logMessage = checkAlerts(thresholdCatalystsAlertPercentage, d_gpuCatalystsMembraneID, h_tempCatalystsString,
					numCatalysts, routerCatalystsQty, (i+1));

				if (!(logMessage.empty()))
					logsFile << logMessage;

			}
			
			catalystsClock--;
			packetsClock--;

			if (catalystsClock == 0) {

				swapCatalystsCPU(numCatalysts, d_gpuCatalystsString, d_gpuCatalystsMembraneID);
				catalystsClock = catalystsClockValue;

			}

			if (packetsClock == 0) {

				cudaTime = (double)clock() / CLOCKS_PER_SEC;

				endCudaTime = endCudaTime + (((double)clock() / CLOCKS_PER_SEC) - cudaTime);

				activatePackets(d_packetsFlags, numPackets, packetsToActivatePercentage);
				packetsClock = packetsClockValue;

			}

			intensiveTrafficIterationsClock--;

			if (intensiveTrafficIterationsClock == 0) {

				char activateSpecificPacketsChoice = 'n';
				bool activateSpecificPackets = false;

				std::cout << "activate one packet iteratively from a specific source (y/n)? ";
				std::cin >> activateSpecificPacketsChoice;

				if (activateSpecificPacketsChoice == 'y' || activateSpecificPacketsChoice == 'Y') {
					activateSpecificPackets = true;
				}

				if (activateSpecificPackets) {

					int numSources;

					std::cout << "how many sources? ";
					std::cin >> numSources;

					for (int i = 0; i < numSources; ++i) {

						char membraneID[2];
						std::cout << "specify source machine: ";
						std::cin >> membraneID;

						size_t len = strlen(membraneID);

						char* nucleusID = new char[len + 2];
						strcpy(nucleusID, membraneID);
						nucleusID[len] = 'n';
						nucleusID[len + 1] = '\0';

						sourceMembranesForIntensiveTraffic.push_back(nucleusID);

					}

				}

				intensiveTrafficIterationsClock = intensiveTrafficIterations;

			}

			cudaTime = (double)clock() / CLOCKS_PER_SEC;

			endCudaTime = endCudaTime + (((double)clock() / CLOCKS_PER_SEC) - cudaTime);

			for (const char* i : sourceMembranesForIntensiveTraffic) {
					activatePacketInNucleus(d_gpuPackets, numPackets, d_packetsFlags, i);
			}

			cudaSyncFile << endCudaTime << std::endl;

		}

		rulesFile.close();

	}
else {

	std::ofstream rulesFile("output/rules_time_cpu.txt", std::ios::trunc);

	rulesFile.close();

	rulesFile.open("output/rules_time_cpu.txt", std::ios::app);

	 for (int i = 0; i < nIter; ++i) {

		 time = (double)clock();
		 time = time / CLOCKS_PER_SEC;

		 computeStepCPU(d_gpuPackets, numPackets, d_gpuRules, numRules, numCatalysts,
			 d_catalystsFlags, d_packetsFlags, h_gpuCatalystsString, h_gpuCatalystsMembraneID);

		 timedif = (((double)clock())) / CLOCKS_PER_SEC - time;

		 rulesFile << timedif << std::endl;

		 if ((i + 1) % checksClock == 0) {

			 std::string logMessage = checkAlerts(thresholdCatalystsAlertPercentage, h_gpuCatalystsMembraneID, h_gpuCatalystsString,
				 numCatalysts, routerCatalystsQty, (i + 1));

			 if (!(logMessage.empty()))
				 logsFile << logMessage;

		 }

		 catalystsClock--;
		 packetsClock--;

		 if (catalystsClock == 0) {

			 swapCatalystsCPU(numCatalysts, h_gpuCatalystsString, h_gpuCatalystsMembraneID);

			 catalystsClock = catalystsClockValue;

		 }

		 if (packetsClock == 0) {

			 activatePackets(d_packetsFlags, numPackets, packetsToActivatePercentage);
			 packetsClock = packetsClockValue;

		 }

		 intensiveTrafficIterationsClock--;

		 if (intensiveTrafficIterationsClock == 0) {

			 char activateSpecificPacketsChoice = 'n';
			 bool activateSpecificPackets = false;

			 std::cout << "activate one packet iteratively from a specific source (y/n)? ";
			 std::cin >> activateSpecificPacketsChoice;

			 if (activateSpecificPacketsChoice == 'y' || activateSpecificPacketsChoice == 'Y') {
				 activateSpecificPackets = true;
			 }

			 if (activateSpecificPackets) {

				 int numSources;

				 std::cout << "how many sources? ";
				 std::cin >> numSources;

				 for (int i = 0; i < numSources; ++i) {
					 
					 char membraneID[2];
					 std::cout << "specify source machine: ";
					 std::cin >> membraneID;

					 size_t len = strlen(membraneID);

					 char* nucleusID = new char[len + 2];
					 strcpy(nucleusID, membraneID);
					 nucleusID[len] = 'n';
					 nucleusID[len + 1] = '\0';

					 sourceMembranesForIntensiveTraffic.push_back(nucleusID);

				 }

			 }

			 intensiveTrafficIterationsClock = intensiveTrafficIterations;

		 }


		 for (const char* i : sourceMembranesForIntensiveTraffic) {
				activatePacketInNucleus(d_gpuPackets, numPackets, d_packetsFlags, i);
		 }

	 }

	 rulesFile.close();
		 
	}

	logsFile.close();

	// just in case
	cudaDeviceSynchronize();

	/* for (int i = 0; i < numPackets; ++i) {
		std::cout << "packet " << i << " in " << d_gpuPackets[i].membraneID << std::endl;
	} */

	// memory cleanup
	for (int i = 0; i < numCatalysts; ++i) {

		if (h_tempCatalystsString[i]) {
			free(h_tempCatalystsString[i]);
			h_tempCatalystsString[i] = nullptr;
		}

		char* d_string;
		cudaMemcpy(&d_string, &h_gpuCatalystsString[i], sizeof(char*), cudaMemcpyDeviceToHost);
		if (d_string) {
			cudaFree(d_string);
		}

		char* d_membraneID;
		cudaMemcpy(&d_membraneID, &h_gpuCatalystsMembraneID[i], sizeof(char*), cudaMemcpyDeviceToHost);
		if (d_membraneID) {
			cudaFree(d_membraneID);
		}

	}

	free(h_gpuCatalystsString);
	free(h_tempCatalystsString);
	free(h_gpuCatalystsMembraneID);

	for (int i = 0; i < numCatalysts; ++i) {
		char* d_string;
		cudaMemcpy(&d_string, &d_gpuCatalystsString[i], sizeof(char*), cudaMemcpyDeviceToHost);
		cudaFree(d_string);

		char* d_membraneID;
		cudaMemcpy(&d_membraneID, &d_gpuCatalystsMembraneID[i], sizeof(char*), cudaMemcpyDeviceToHost);
		cudaFree(d_membraneID); 
	}

	cudaFree(d_gpuCatalystsString);
	cudaFree(d_gpuCatalystsMembraneID);

	for (char* membrane : sourceMembranesForIntensiveTraffic) {
		delete[] membrane;
	}

	cudaFree(d_gpuPackets);
	cudaFree(d_gpuRules);
	cudaFree(d_catalystsFlags);
	cudaFree(d_packetsFlags);

	for (size_t i = 0; i < numPackets; ++i) {
		GPUPacket& gpuPacket = gpuPackets[i];

		if (gpuPacket.membraneID) {
			delete[] gpuPacket.membraneID;
		}

	}

	for (int i = 0; i < numRules; ++i) {
		GPURule& gpuRule = gpuRules[i];
		if (gpuRule.cond1) delete[] gpuRule.cond1;
		if (gpuRule.cond2) delete[] gpuRule.cond2;
		if (gpuRule.result1) delete[] gpuRule.result1;
		if (gpuRule.result2) delete[] gpuRule.result2;
		delete[] gpuRule.destination;
		delete[] gpuRule.membraneID;
	}

	for (int i = 0; i < numCatalysts; ++i) {
		GPUCatalyst& gpuCatalyst = gpuCatalysts[i];
		delete[] gpuCatalyst.string;
		delete[] gpuCatalyst.membraneID;
	}

	gpuPackets.clear();
	gpuRules.clear();
	gpuCatalysts.clear();

	delete cell.rootMembrane;

	return 0;

}