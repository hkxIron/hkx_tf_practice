#include <iostream>
#include <algorithm>
// blog: https://blog.csdn.net/theonegis/article/details/45801201
using namespace std;

/************************************************************************/
/* 数塔问题                                                               */
/************************************************************************/
const int N = 50;//为了算法写起来简单，这里定义一个足够大的数用来存储数据(为了避免运算过程中动态申请空间，这样的话算法看起来比较麻烦，这里只是为了算法看起来简单)
int data[N][N]={
    {9},
    {12,15},
    {10,6,8},
    {2,18,9,5},
    {19,7,10,4,16}
};//存储数塔原始数据
int n=5;//塔的层数
int tower_sum[N][N];//存储动态规划过程自底向上的求和

/*动态规划实现数塔求解*/
void tower_walk()
{
    // tower_sum初始化
    for (int i = 0; i < n; ++i)
    {
        tower_sum[n - 1][i] = data[n - 1][i]; // 最底层的直接赋值
    }

    int temp_max;
    for (int i = n - 1; i >= 0; --i) // i 指层数
    {
        for (int j = 0; j <= i; ++j)
        {
            // 使用递推公式计算tower_sum的值
            temp_max = std::max(tower_sum[i + 1][j], tower_sum[i + 1][j + 1]);
            tower_sum[i][j] = temp_max + data[i][j]; // 更新当前层的tower_sum
        }
    }
}

/*打印最终结果*/
void print_result()
{
    cout << "最大路径和：" << tower_sum[0][0] << '\n';
    int node_value;
    // 首先输出塔顶元素
    cout << "最大路径：" << data[0][0];
    int j = 0;
    for (int i = 1; i < n; ++i)
    {
        node_value = tower_sum[i - 1][j] - data[i - 1][j];
        /* 如果node_value == tower_sum[i][j]则说明下一步应该是data[i][j]；如果node_value == tower_sum[i][j + 1]则说明下一步应该是data[i][j + 1]*/
        if (node_value == tower_sum[i][j + 1]) ++j;
        cout << "->" << data[i][j];
    }
    cout << endl;
}

int main()
{
    /*
    cout << "输入塔的层数：";
    cin >> n;
    cout << "输入塔的节点数据(第i层有i个节点)：\n";
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            cin >> data[i][j];
        }
    }
    */

    tower_walk();
    print_result();
}