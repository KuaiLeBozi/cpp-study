# leetcode3

```cpp
//初始代码
class Solution {
    public:
        int longestConsecutive(vector<int>& nums) {
            sort(nums.begin(),nums.end());
            int sum=1;
            int temp=1;
            for(int i=0;i<nums.size()-1;i++){
                if(nums[i]+1!=nums[i+1]){
                    if(nums[i]==nums[i+1]) continue;
                    sum = max(sum, temp);
                    temp=0;
                }
                temp++;
            }
            return max(sum,temp);
        }
    };//一开始忘记最后如果一直连续下去，sum得不到更新的机会，故最后return依然要带上temp；同时对于数据元素相等的情况没有充分理解题目，相等的时候temp不用置为0
```

本意比较简单,但有错误

**sort()基本使用方法**

​	sort()函数可以对给定区间所有元素进行排序。它有三个参数sort(begin, end, **cmp**)，其中begin为指向待sort()的数组的第一个元素的指针，end为指向待sort()的数组的最后一个元素的下一个位置的指针，**cmp参数为排序准则**，cmp参数**可以不写**，如果不写的话，默认**从小到大**进行排序。如果我们想从大到小排序可以将cmp参数写为greater<int>()就是对int数组进行排序，当然<>中我们也可以写double、long、float等等。如果我们需要按照其他的排序准则，那么就需要我们自己定义一个bool类型的函数来传入。

```cpp
bool cmp(int x,int y){
	return x % 10 > y % 10;
}
```

---

#### 哈希表解法

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> num_set;
        for (const int& num : nums) {
            num_set.insert(num);
        }

        int longestStreak = 0;//不能设为1，因为还有空集的情况！！！

        for (const int& num : num_set) {//unordered_set元素是const类型，要用常量引用
            if (!num_set.count(num - 1)) {//.count是统计出现次数
                int currentNum = num;
                int currentStreak = 1;

                while (num_set.count(currentNum + 1)) {//完美解决了元素重叠的问题，只用看value
                    currentNum += 1;
                    currentStreak += 1;
                }

                longestStreak = max(longestStreak, currentStreak);
            }
        }

        return longestStreak;           
    }
};//利用set作为哈希表
```

主要看其中是怎么避免元素重复的