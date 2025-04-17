# leetcode4

```cpp
//自己的解法，用的双指针
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int slowIndex = 0;
        for(int fastIndex = 0; fastIndex<nums.size(); fastIndex++){/*最好把nums.size()换成临时变量*/
            if(nums[fastIndex]!=0) swap(nums[fastIndex],nums[slowIndex++]);
        }
    }
};
```

---

#### 方法一：双指针

思路及解法

使用双指针，左指针指向当前已经处理好的序列的尾部，右指针指向待处理序列的头部。

右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移。

注意到以下性质：

```tex
1 左指针左边均为非零数；

2 右指针左边直到左指针处均为零。
```

因此每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变。

```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int n = nums.size(), left = 0, right = 0;
        while (right < n) {
            if (nums[right]) {
                swap(nums[left], nums[right]);
                left++;
            }
            right++;
        }
    }
};
```

