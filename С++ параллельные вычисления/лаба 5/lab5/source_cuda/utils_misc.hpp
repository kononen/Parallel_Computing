#pragma once

#include <string>
#include <iostream>

// ------------
// --- MISC ---
// ------------
inline void exit_with_error(const std::string &msg) {
	std::cout << "ERROR: " << msg << "\n";
	exit(1);
}

inline std::string bool_to_str(bool arg) {
	return arg ? "TRUE" : "FALSE";
}
