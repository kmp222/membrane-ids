#include <vector>
#include <string>
#include <iostream>

struct Rule {

	std::vector<std::string> conditions;
	std::vector<std::string> result;

	std::string destination;

	Rule(const std::vector<std::string>& cond, const std::vector<std::string>& res,
		const std::string& dst)
		: conditions(cond), result(res), destination(dst) {}

};