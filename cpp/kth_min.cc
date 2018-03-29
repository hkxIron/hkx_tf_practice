// blog:https://github.com/hkxIron/algorithm/blob/master/sword_offer/src/034.cpp
/*
	[最小k个数]

    [题目]
	输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
    [解析]
    方法0: 直接排序然后返回前k个，最好的时间复杂度为 O(nlog(n))
    方法1: 快排的变种，时间复杂度 O(n)，缺点：原址，需要把所有数都 load 到内存中
    方法2: 利用最大堆作为辅助，时间复杂度 O(n*lg(k))，适用于处理数据量很大的情况。
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

        if (method=="heap"){
            return GetLeastNumbersHeap(input, k);
        }else{
            GetLeastNumbersPartition(input, 0, input.size()-1, k);
            for(int i=0; i<k; i++) ans.push_back(input[i]);
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
        }else if (pos < k-1){ // 它与快排的区别是,它只会使某一边的部分有序
            GetLeastNumbersPartition(input, pos+1, right, k); // 第k大在右边
        }else{
            GetLeastNumbersPartition(input, left, pos-1, k); // 第k大在左边
        }
    }



    /**
    *  交换顺序表a中的区间[low,hight]中的记录，枢轴记录到位，并返回其所在的位置，
    *  此时，在它之前的记录均小于它，在它之后的均大于它
       这个是数据结构一书中的划分·
    */
    int partition_book(vector<int> &a, int low, int high){
        int pivot = a[low] ;
        while(low<high){
            while(low<high&&a[high]>=pivot) --high; // 将右边小于pivot的找出来
            a[low] = a[high];
            while(low<high&&a[low]<=pivot) ++low; // 将左边大于pivot的找出来
            a[high] = a[low];
        }
        a[low] = pivot;
        return low; // 此时low与high相等
    }

    // 这个是本程序原始的划分
    int partition(vector<int> &input, int left, int right){
        if(left > right)
            return -1;
        int pos = left-1;
        for(int i=left; i<right; i++){
            if(input[i] <= input[right]){
                std::swap(input[i], input[++pos]);
            }
        }

        std::swap(input[right], input[++pos]);
        // input[left, pos] <= input[pos]
        // input[pos+1, right] > input[pos]
        return pos;
    }

    // heap sort, time complexity - O(nlog(k))
    vector<int> GetLeastNumbersHeap(vector<int> &input, int k){

        if(k > input.size() || input.empty())
            return vector<int>();

        // 建立最大堆(所有的元素都比堆顶元素小)
        vector<int> ans(input.begin(), input.begin()+k); // max heap, 只有k个元素
        make_heap(ans.begin(), ans.end(), comp); // 将这k个元素建立一个堆

        for(int i=k; i<input.size(); i++){
            // 若当前元素比堆顶元素小，而堆里只有k个元素，因此当前元素可能是前k小中的一个，则入堆
            if(input[i] < ans.front()){ // the current value less than the maximun of heap
                pop_heap(ans.begin(), ans.end(), comp); // 弹出heap顶元素, 将其放置于区间末尾. O(logN)
                ans.pop_back(); //

                ans.push_back(input[i]);// 将当前元素入堆
                push_heap(ans.begin(), ans.end(), comp);
            }
        }

        sort(ans.begin(), ans.end());// 对堆调整

        return ans;
    }

    static bool comp(int a, int b){
        return a<b;
    }

};

int main()
{
    vector<int> input={1,-3,8,9,3,0,-2,10,-4,7,20,-21,3,10};
    /*freopen("in.txt", "r", stdin);
    cin >> k;
    int cur;
    while(cin >> cur){
        input.push_back(cur);
    }
    */
    int k=6;
    vector<int> ans = Solution().GetLeastNumbers_Solution(input, k,"part");

    cout<<k<<"-th min:";
    for(int n : ans) cout << n << " "; cout << endl; // 注意，此处ans只是部分有序
    cout<<"array after partition:"<<endl;
    copy(input.begin(), input.end(), std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    cout<<endl;
    //-----------------------
    {
       std::sort (input.begin(), input.end());
       cout<<"standard result:"<<endl;
       //for(auto n : input) cout << n << " "; cout << endl;
       copy(input.begin(), input.end(), std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    }

    //fclose(stdin);
    return 0;
}
