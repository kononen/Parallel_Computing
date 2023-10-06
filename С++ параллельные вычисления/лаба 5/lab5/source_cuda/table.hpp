#pragma once

#include <iostream>
#include <iomanip>
#include <string>


const std::streamsize COL_SIZE_1 = 8;

template<typename Object>
void table_add_1(const Object &obj) {
	std::cout
		<< " | "
		<< std::setw(COL_SIZE_1) << std::fixed << obj
		<< " | " << std::flush;
}


const std::streamsize COL_SIZE_2 = 10;
const std::streamsize COL_2_PRECISION = 2;

template<typename Object>
void table_add_2(const Object &obj) {
	std::cout
		<< std::setw(COL_SIZE_2) << std::fixed << std::setprecision(COL_2_PRECISION) << obj
		<< " | " << std::flush;
}


const std::streamsize COL_SIZE_3 = 7;
const std::streamsize COL_3_PRECISION = 2;

template<typename Object>
void table_add_3(const Object &obj) {
	std::cout
		<< std::setw(COL_SIZE_3) << std::fixed << std::setprecision(COL_3_PRECISION) << obj
		<< " |\n" << std::flush;
}


void table_hline() {
	std::string str1("");
	std::string str2("");
	std::string str3("");
	str1.insert(0, 1 + COL_SIZE_1 + 1, '-');
	str2.insert(0, 1 + COL_SIZE_2 + 1, '-');
	str3.insert(0, 1 + COL_SIZE_3 + 1, '-');

	std::cout << " |" << str1 << "|" << str2 << "|" << str3 << "|\n" << std::flush;
}