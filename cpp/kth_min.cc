/*
g++ -std=c++11 -o kth_min kth_min.cc &&./kth_min
*/

// blog:https://github.com/hkxIron/algorithm/blob/master/sword_offer/src/034.cpp
/*
	[最小k个数]

    [题目]
	输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
    [解析]
    方法0: 直接排序然后返回前k个，最好的时间复杂度为 O(nlog(n))
    方法1: 快排的变种，时间复杂度 O(n)，缺点：原址，需要把所有数都 load 到内存中
    方法2: 利用 最大堆 作为辅助，时间复杂度 O(n*log(k))，适用于处理数据量很大的情况。
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace std;

class Solution{
public:
    vector<int> GetLeastNumbers_Solution(vector<int>& input, int k, string method="heap"){
        vector<int> ans;
        if(k > input.size()){
            return ans;
        }
        cout<<"method:"+method<<endl;
        if (method=="heap"){
            return GetLeastNumbersHeap(input, k);
        }else{
            GetLeastNumbersPartition(input, 0, input.size()-1, k);
            for(int i=0; i<k; i++) ans.push_back(input[i]); // 数据的前k个即 最小的k个数
            return ans;
        }
    }

    // parition, average time complexity - O(n)
    // note: k should less than then size of input
    void GetLeastNumbersPartition(vector<int> &input, int left, int right, int k){
        int pos = partition_book(input, left, right);
        //int pos = partition(input, left, right);
        if(pos == k-1){
            return;
        }else if (pos < k-1){ // 它与快排的区别是,它只会找到第k个后停止
            // 注意,此法虽然找到前k个元素,但这k个元素之间并不一定有序
            GetLeastNumbersPartition(input, pos+1, right, k); // 第k大在右边
        }else{
            GetLeastNumbersPartition(input, left, pos-1, k); // 第k大在左边
        }
    }



    /**
    *  交换顺序表a中的区间[low,hight]中的记录，枢轴记录到位，并返回其所在的位置，
    *  此时，在它之前的记录均小于它，在它之后的均大于它
       这个是《数据结构》一书中的划分·
    */
    int partition_book(vector<int> &a, int low, int high){
        int pivot = a[low] ;
        // 一直进行交换,直到左右指针相遇
        while(low<high){
            while(low<high&&a[high]>=pivot) --high; // 将右边小于pivot的找出来, 并放到左边
            a[low] = a[high];
            while(low<high&&a[low]<=pivot) ++low; // 将左边大于pivot的找出来,并放到右边，此时low所在的位置空缺了
            a[high] = a[low];
        }
        a[low] = pivot; // 找到pivot的位置
        return low; // 此时low与high相等,也即是pivot的索引值
    }

    /**
	### **另一个划分例子演示**
	我们用一个更典型的例子来展示 `partition` 函数如何工作。假设输入数组为 `[10, 80, 30, 90, 40, 50, 70]`，选择最右元素 `70` 作为基准值（pivot）。
	
	#### **初始状态**
	```
	数组: [10, 80, 30, 90, 40, 50, 70]
	区间: left = 0, right = 6
	pivot = 70
	pos = left - 1 = -1
	```
	
	#### **遍历过程（i 从 left 到 right-1）**
	1. **i=0**：`input[0] = 10` ≤ `70`  
	   - `pos++` → `pos=0`  
	   - 交换 `input[0]` 和 `input[0]`（无变化）  
	   - 数组状态: `[10, 80, 30, 90, 40, 50, 70]`  
	
	2. **i=1**：`input[1] = 80` > `70`  
	   - 不操作，跳过  
	   - 数组状态不变  
	
	3. **i=2**：`input[2] = 30` ≤ `70`  
	   - `pos++` → `pos=1`  
	   - 交换 `input[2]` 和 `input[1]`  
	   - 数组状态: `[10, 30, 80, 90, 40, 50, 70]`  
	
	4. **i=3**：`input[3] = 90` > `70`  
	   - 不操作，跳过  
	   - 数组状态不变  
	
	5. **i=4**：`input[4] = 40` ≤ `70`  
	   - `pos++` → `pos=2`  
	   - 交换 `input[4]` 和 `input[2]`  
	   - 数组状态: `[10, 30, 40, 90, 80, 50, 70]`  
	
	6. **i=5**：`input[5] = 50` ≤ `70`  
	   - `pos++` → `pos=3`  
	   - 交换 `input[5]` 和 `input[3]`  
	   - 数组状态: `[10, 30, 40, 50, 80, 90, 70]`  
	
	#### **放置基准值**
	- `pos++` → `pos=4`  
	- 交换 `input[right]`（即 `input[6]=70`）和 `input[4]`  
	- 最终数组: `[10, 30, 40, 50, 70, 90, 80]`  
	
	#### **划分结果**
	- **左半部分**：`[10, 30, 40, 50]`（所有元素 ≤ `70`）  
	- **右半部分**：`[90, 80]`（所有元素 > `70`）  
	- **基准值位置**：`pos=4`  
	
	---
	
	### **关键点验证**
	1. **`pos` 的作用**：始终指向“已处理的 ≤ pivot 部分”的末尾。  
	2. **交换逻辑**：将 ≤ pivot 的元素逐步交换到左侧。  
	3. **最终交换**：将 pivot 放到正确位置（`pos+1`）。  
	*/
    int partition(vector<int> &input, int left, int right){
        if(left > right)
            return -1;
		// 将input[right]当成pivot元素
		// pos 的作用是标记当前已处理区间中所有 ≤ pivot 的元素的右边界
        int pos = left-1;
        for(int i=left; i<right; i++){
            if(input[i] <= input[right]){
                std::swap(input[i], input[++pos]);
            }
        }
		// 将pivot放到 ++pos
        std::swap(input[right], input[++pos]);
        // input[left, pos] <= input[pos]
        // input[pos+1, right] > input[pos]
        return pos;
    }

    // heap sort, time complexity - O(nlog(k))
    vector<int> GetLeastNumbersHeap(vector<int> &input, int k){

        if(k > input.size() || input.empty())
            return vector<int>();

        // 建立最大堆(所有的元素都比堆顶元素小)， 注意，堆在出堆时才保证出堆的元素有序，在建堆时并不保证所有元素有序，只保证部分元素有序
        vector<int> ans(input.begin(), input.begin()+k); // max heap, 只有k个元素
        make_heap(ans.begin(), ans.end(), comp); // 将这k个元素建立一个堆

        // 从第k+1个元素开始遍历
        for(int i=k; i<input.size(); i++){
            // 若当前元素比堆顶元素小(最大堆的堆顶元素是堆里最大的元素)，而堆里只有k个元素，因此当前元素可能是前k小中的一个，则入堆
            if(input[i] < ans.front()){ // the current value less than the maximun of heap
                pop_heap(ans.begin(), ans.end(), comp); // 弹出heap顶元素, 将其放置于区间末尾. O(logN)
                ans.pop_back(); // 弹出

                ans.push_back(input[i]);// 将当前元素入堆
                push_heap(ans.begin(), ans.end(), comp);
            }
        }

        //sort(ans.begin(), ans.end());// 为了使元素间有序,但并非必要的步骤

        return ans;
    }

    static bool comp(int a, int b){
        return a<b;
    }

};

int main()
{
    /*freopen("in.txt", "r", stdin);
    cin >> k;
    int cur;
    while(cin >> cur){
        input.push_back(cur);
    }
    */
    {
        vector<int> input={1,-3,8,9,3,0,-2,10,-4,7,20,-21,3,10, -4,-7};
        int k=6;
        vector<int> ans = Solution().GetLeastNumbers_Solution(input, k,"partition");

        cout<<k<<"-th min:";
        for(int n : ans) cout << n << " "; cout << endl; // 注意，此处ans只是部分有序
        cout<<"array after partition:"<<endl;
        copy(input.begin(), input.end(), std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
        cout<<endl<<endl;
    }
    {
        vector<int> input={1,-3,8,9,3,0,-2,10,-4,7,20,-21,3,10, -4,-7};
        int k=6;
        vector<int> ans = Solution().GetLeastNumbers_Solution(input, k,"heap");

        cout<<k<<"-th min:";
        for(int n : ans) cout << n << " "; cout << endl; // 注意，此处ans只是部分有序
        cout<<"array after partition:"<<endl;
        copy(input.begin(), input.end(), std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
        cout<<endl<<endl;
    }
    //-----------------------
    {
       vector<int> input={1,-3,8,9,3,0,-2,10,-4,7,20,-21,3,10, -4,-7};
       std::sort(input.begin(), input.end());
       cout<<"standard result:"<<endl;
       //for(auto n : input) cout << n << " "; cout << endl;
       copy(input.begin(), input.end(), std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    }

    //fclose(stdin);
    return 0;
}




