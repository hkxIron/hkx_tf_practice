/*******************************************************************
Copyright(c) 2016, Harry He
All rights reserved.
Distributed under the BSD license.
(See accompanying file LICENSE.txt at
https://github.com/zhedahht/CodingInterviewChinese2/blob/master/LICENSE.txt)
*******************************************************************/

//==================================================================
// 《剑指Offer——名企面试官精讲典型编程题》代码
// 作者：何海涛
//==================================================================

// 面试题44：数字序列中某一位的数字
// 题目：数字以0123456789101112131415…的格式序列化到一个字符序列中。在这
// 个序列中，第5位（从0开始计数）是5，第13位是1，第19位是4，等等。请写一
// 个函数求任意位对应的数字。

#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

int countOfIntegers(int digits);
int digitAtIndex(int index, int digits);
int beginNumber(int digits);

int digitAtIndex(int index)
{
	if(index < 0)
		return -1;

	int digits = 1; // 当前的数字有多少位
	while(true)
	{
		int numbers = countOfIntegers(digits); // 当前位数的数字有多少个
		if(index < numbers * digits)
			return digitAtIndex(index, digits);

		index -= digits * numbers;
		digits++;
	}

	return -1;
}

// digits: 当前段内数字的位数, 3
// index: 当前段内的char偏移量, 811
// 接下来的2700位是900个100~900的三位数，由于811<2700，所以第811位是某个三位数中的一位，
// 由于811 =270*3 +1, 说明第811位是从100开始的第270个数字即370的中间一位，也就是7
int digitAtIndex(int index, int digits)
{
	int number = beginNumber(digits) + index / digits; // 100+811/3 = 370
	int indexFromRight = digits - index % digits; // 3 - 811%3 = 3 -1 =2
	for(int i = 1; i < indexFromRight; ++i)
		number /= 10;
	return number % 10;
}

// 输入2，则返回两位数的个数90(10~90)
// 输入3，则返回三位数的个数900(100~999)
int countOfIntegers(int digits)
{
	if(digits == 1)
		return 10;

	int count = (int)pow(10, digits - 1);
	return 9 * count;
}

// 第一个两位数100,第一个三位数为100
int beginNumber(int digits)
{
	if(digits == 1)
		return 0;

	return (int)pow(10, digits - 1);
}

// ====================测试代码====================
void test(const char* testName, int inputIndex, int expectedOutput)
{
	if(digitAtIndex(inputIndex) == expectedOutput)
		cout << testName << " passed." << endl;
	else
		cout << testName << " FAILED." << endl;
}


int main()
{
	test("Test1", 0, 0);
	test("Test2", 1, 1);
	test("Test3", 9, 9);
	test("Test4", 10, 1);
	test("Test5", 189, 9);  // 数字99的最后一位，9
	test("Test6", 190, 1);  // 数字100的第一位，1
	test("Test7", 1000, 3); // 数字370的第一位，3
	test("Test8", 1001, 7); // 数字370的第二位，7
	test("Test9", 1002, 0); // 数字370的第三位，0
	return 0;
}
