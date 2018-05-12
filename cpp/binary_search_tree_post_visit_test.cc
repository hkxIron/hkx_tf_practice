/*
	[二叉搜索树的后序遍历序列]

    [题目]
	输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
	如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
    [解析]
    理解二叉搜索树的定义。


    二叉搜索树(英语：Binary Search Tree)，也称二叉查找树、有序二叉树(英语：ordered binary tree)，排序二叉树（英语：sorted binary tree），是指一棵空树或者具有下列性质的二叉树：

    任意节点的左子树不空，则左子树上所有结点的值均小于它的根结点的值；
    任意节点的右子树不空，则右子树上所有结点的值均大于它的根结点的值；
    任意节点的左、右子树也分别为二叉查找树；
    没有键值相等的节点。

    注意：
        - 任意两个数互不相同。
        - 代码实现中数组为空的情况要考虑在内。


     解析：
     然后根据一个具体的实例模拟一下过程：
例如输入数组{5、7、6、9、11、10、8}

因为是后序遍历，所以最后一个数字8一定是整棵树的根结点。所以其他数字为8的子结点，
从前往后遍历数组，5，7，6都小于8，所以这三个数字应该组成了8的左子树，同理9、11、10组成了8的右子树。
*/

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution{
public:
    bool VerifySquenceOfBST(vector<int> sequence){
        if(sequence.empty()) // Note: special case
            return false;

        return VerifySquenceOfBSTRecursive(sequence, 0, sequence.size()-1);
    }

    bool VerifySquenceOfBSTRecursive(vector<int> &sequence, int left, int right){
        // 如果为空或者只有一个节点，返回true
        if(left >= right) // null tree or just have one note
            return true;

        // 得到根结点
        int rootVal = sequence[right];
        // 查找右子树的最左边元素
        int iright = right-1;
        while(iright >= left && sequence[iright] > rootVal) iright--;

        // check left tree, the value should be less than rootVal
        // 检查左子树是否符合搜索二叉树的定义，即都小于根结点
        int ileft = iright-1;
        for( ; ileft>=left; ileft--){
            if(sequence[ileft] > rootVal)
                return false;
        }

        return VerifySquenceOfBSTRecursive(sequence, left, iright-1)
            && VerifySquenceOfBSTRecursive(sequence, iright, right-1);
    }
};

int main()
{
    /*
           12
         /   \
        10    15
       / \   /   \
      9  11  13  16

      后序遍历： 9,11,10,13,16,15,12

    */
    vector<int> v = {9,11,10,13,16,15,12}


    return 0;
}