# leetcode9

```cpp
//我的代码，失败版本（居然要考虑复数啊啊啊啊啊啊啊啊想用滑动窗口，滑动窗口需要满足单调性）
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int leftIndex = 0;
        int rightIndex = 0;
        int sum = 0, result = 0;
        
        int length = nums.size();
        while(leftIndex < length){
            sum += nums[rightIndex++];
            if(sum > k){
                sum -= nums[leftIndex++];
            }
            if(sum == k && leftIndex < length){result++;}
            if(rightIndex >= length){
                leftIndex++;
                rightIndex = leftIndex;
                sum = 0;
            }
        }
        return result;

    }
};
```

---

#### 方法一：枚举

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int count = 0;
        for (int start = 0; start < nums.size(); ++start) {
            int sum = 0;
            for (int end = start; end >= 0; --end) {
                sum += nums[end];
                if (sum == k) {
                    count++;//一方面从start往前，一方面边遍历边算结果，减少时间复杂度
                }
            }
        }
        return count;
    }
};
```

考虑以 i 结尾和为 k 的连续子数组个数，我们需要统计符合条件的下标 j 的个数，其中 0≤j≤i 且 [j..i] 这个子数组的和恰好为 k 。

我们可以枚举 [0..i] 里所有的下标 j 来判断是否符合条件，可能有读者会认为假定我们确定了子数组的开头和结尾，还需要 O(n) 的时间复杂度遍历子数组来求和，那样复杂度就将达到 O(n3) 从而无法通过所有测试用例。但是如果我们知道 [j,i] 子数组的和，就能 O(1) 推出 [j−1,i] 的和，因此这部分的遍历求和是不需要的，我们在枚举下标 j 的时候已经能 O(1) 求出 [j,i] 的子数组之和。

---

#### 方法二：前缀和 + 哈希表优化

思路和算法

我们可以基于方法一利用数据结构进行进一步的优化，我们知道方法一的瓶颈在于对每个 i，我们需要枚举所有的 j 来判断是否符合条件，这一步是否可以优化呢？答案是可以的。

我们定义 pre[i] 为 [0..i] 里所有数的和，则 pre[i] 可以由 pre[i−1] 递推而来，即：
pre[i]=pre[i−1]+nums[i]

那么「[j..i] 这个子数组和为 k 」这个条件我们可以转化为
pre[i]−pre[j−1]==k

简单移项可得符合条件的下标 j 需要满足
pre[j−1]==pre[i]−k

所以我们考虑以 i 结尾的和为 k 的连续子数组个数时只要统计有多少个前缀和为 pre[i]−k 的 pre[j] 即可。我们建立哈希表 mp，以和为键，出现次数为对应的值，记录 pre[i] 出现的次数，从左往右边更新 mp 边计算答案，那么以 i 结尾的答案 mp[pre[i]−k] 即可在 O(1) 时间内得到。最后的答案即为所有下标结尾的和为 k 的子数组个数之和。

需要注意的是，从左往右边更新边计算的时候已经保证了mp[pre[i]−k] 里记录的 pre[j] 的下标范围是 0≤j≤i 。同时，由于pre[i] 的计算只与前一项的答案有关，因此我们可以不用建立 pre 数组，直接用 pre 变量来记录 pre[i−1] 的答案即可。

下面的动画描述了这一过程：

![img](https://assets.leetcode-cn.com/solution-static/560/1.PNG)

![img](https://assets.leetcode-cn.com/solution-static/560/2.PNG)

![img](https://assets.leetcode-cn.com/solution-static/560/3.PNG)

![img](https://assets.leetcode-cn.com/solution-static/560/4.PNG)

![img](https://assets.leetcode-cn.com/solution-static/560/5.PNG)

![img](https://assets.leetcode-cn.com/solution-static/560/6.PNG)

![img](https://assets.leetcode-cn.com/solution-static/560/7.PNG)

![img](https://assets.leetcode-cn.com/solution-static/560/8.PNG)

![img](https://assets.leetcode-cn.com/solution-static/560/9.PNG)

```cpp
class Solution {//啊代码量折磨少？？？？
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        mp[0] = 1;
        int count = 0, pre = 0;
        for (auto& x:nums) {
            pre += x;
            if (mp.find(pre - k) != mp.end()) {
                count += mp[pre - k];//注意这里可能因为从该位置往前不止一种连续数组满足情况
            }
            mp[pre]++;//双重功能，可创建可叠加。
        }
        return count;
    }
};
```

复杂度分析

    时间复杂度：O(n)，其中 n 为数组的长度。我们遍历数组的时间复杂度为 O(n)，中间利用哈希表查询删除的复杂度均为 O(1)，因此总时间复杂度为 O(n)。
    
    空间复杂度：O(n)，其中 n 为数组的长度。哈希表在最坏情况下可能有 n 个不同的键值，因此需要 O(n) 的空间复杂度。

**前缀和的知识**！！！！