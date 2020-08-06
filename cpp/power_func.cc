/*
    blog: https://github.com/hkxIron/algorithm/blob/master/sword_offer/src/059.cpp
	[数值的整数次方]

    [题目]
	给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方(尽可能高效快速)。
    [解析]
    主要是要注意边界条件的处理。
*/

#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <limits>

using namespace std;

class Solution{
public:
    double Power(double base, int exponent) {
        // special case
        if(base == 0 && exponent == 0)
            return 0;
        if(base == 0 && exponent < 0 ) //负无穷
            return numeric_limits<int>::max();

        double ans = PowerPositive(base, abs(exponent));
        if(exponent < 0)
            ans = 1.0/ans;
        return ans;
    }

    /**
    *  递归计算方式
    */
    double PowerPositive(double base, int exponent){
        if(exponent == 0) return 1;
        if(exponent == 1) return base;

        // x^5= x^2 * x^2 *x
        // x^4 = x^2 * x^2
        double temp = PowerPositive(base, exponent/2);
        bool is_even = (exponent&1) == 1; // 是否是奇数
        if(is_even){
            return temp*temp*base;
        }else{
            return temp*temp;
        }
    }

    /**
     * 非递归计算方式
     * 
     * 1.全面考察指数的正负、底数是否为零等情况。
     * 2.写出指数的二进制表达，例如13的二进制, 13=8+4+0+1 = 1101。
     * 3.举例10的13次方:10^13=10^(1101) = 10^0001*10^0100*10^1000。
     * 4.通过&1和>>1来逐位读取1101，为1时将该位代表的乘数累乘到最终结果。

     5=4 + 1 = 101
     3^5 = 3^(100)* 3^(001) =(3^4)*(3^1)
     */
    double PowerPositiveNonRecursive(double base, int n){
        if(n == 0) return 1;
        if(n == 1) return base;
        int exponent =abs(n);
	
	// 2^5, curr_base=2, exponent=5=4+0+1 = 101 
	// exponent最低位为1, curr_base = 2 , res=2 , curr_base=4, exponent右移1位= 2 = 10
	// exponent最低位为0, curr_base =4, res不变,curr_base=16, exponent右移1位= 1
	// exponent最低位为1, curr_base =16, res=res*16=32, exponent右移1位= 0
	// 退出
        double res = 1,curr_base = base;
	// exponent一直右移除2    
        while(exponent!=0){
            if((exponent&1)==1) // 当前最低为1
                res*=curr_base;// 有1时，就将结果乘进去

            curr_base*=curr_base;// 当前的乘以的基数翻倍，而不是res翻倍
            exponent>>=1;// 右移一位,除2
        }
        return n>=0?res:1/res;
    }
};

int main()
{
    Solution s;
    double number =2;
    int exp = 10;
    double result = s.Power(number,exp);
    cout<<"ground truth: "<<pow(number,exp)<<endl;
    cout<<"my power func: "<<result<<endl;
    cout<<"my power func: "<<s.PowerPositiveNonRecursive(number,exp)<<endl;
    return 0;
}
