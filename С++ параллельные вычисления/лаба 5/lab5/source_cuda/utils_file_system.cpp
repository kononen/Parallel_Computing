#include "utils_file_system.h"

#include <filesystem>
#include <fstream>

// -----------------------
// --- File operations ---
// -----------------------
void ensure_folder(const std::string &folder) {
	if (!std::filesystem::exists(folder)) std::filesystem::create_directory(folder);
}

void clean_folder(const std::string &folder) {
	for (const auto &entry : std::filesystem::directory_iterator(folder))
		std::filesystem::remove_all(entry.path());
}

void create_empty_files(const std::vector<std::string> &filenames) {
	for (const auto &filename : filenames) {
		std::ofstream outFile(filename);
	}
}