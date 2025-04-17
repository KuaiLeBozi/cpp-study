# leetcode6

```cpp
//本人错误解法
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        auto slowIndex = nums.begin();
        int sum = 0;
        vector<vector<int>> result;
        for(auto fastIndex = slowIndex + 2; fastIndex < nums.end(); fastIndex++){
            int temp = accumulate(slowIndex, fastIndex+1, 0);//注意accumulate函数，最后0是累加到0上，同时注意fastIndex要+1不然不包括fastIndex所指元素，因为左闭右开，不包括结束迭代器
            if(temp == 0) result.push_back(vector<int>(slowIndex,fastIndex+1));//迭代器左闭右开
            slowIndex++;
        }
        return result;
    }
};//只能判断连续的数，题目理解错误，数字可以跳跃的加
```

---

#### 排序+双指针（很难，多看）

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());//先排序
        vector<vector<int>> ans;
        // 枚举 a
        for (int first = 0; first < n; ++first) {
            // 需要和上一次枚举的数不相同（理解为什么不能重复）
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int third = n - 1;
            int target = -nums[first];
            // 枚举 b
            for (int second = first + 1; second < n; ++second) {
                // 需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target) {//注意是大于号而不是!=
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加(精妙之处，没必要重置右指针)
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third) {//要根据前面的判断条件，考虑全面
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    ans.push_back({nums[first], nums[second], nums[third]});//注意这里语法
                }
            }
        }
        return ans;
    }
};
```

* 要看懂题目，要是想防止重复，要么哈希表（增加空间复杂度），要么把数组排序
* 根据排序后的有序关系，可以让c指针从右开始，使第二重循环与第三重循环**并列**
