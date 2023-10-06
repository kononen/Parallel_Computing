#pragma once

#include <vector>
#include <string>

// -----------------------
// --- File operations ---
// -----------------------
void ensure_folder(const std::string &folder);

void clean_folder(const std::string &folder);

void create_empty_files(const std::vector<std::string> &filenames);