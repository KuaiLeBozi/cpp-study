# leetcode7

```cpp
//本人代码
class Solution {//边界问题很烦人
public:
    int trap(vector<int>& height) {
        int sum = 0;
        int slowIndex = 0;
        while(slowIndex < height.size() && !height[slowIndex]){
            slowIndex++;
        }
        if(slowIndex >= height.size()) return sum;
        int temp = 0;
        for(int fastIndex = slowIndex; fastIndex < height.size() || temp != 0; fastIndex++){
            if(fastIndex >= height.size() && temp != 0){
                slowIndex++;
                if(slowIndex >= height.size() - 2) break;
                fastIndex = slowIndex;
                temp = 0;
            }//用于处理右边无更高的值的情况
            if(height[fastIndex] < height[slowIndex]){
                temp += height[slowIndex] - height[fastIndex];//核心思想
            }
            else{
                slowIndex = fastIndex;
                sum += temp;
                temp = 0;
            }
        }
        if(temp != 0){
            temp -= (height[slowIndex - 1] - height[height.size() - 1]) * (height.size() - slowIndex);
            sum += temp;
        }
        return sum;
    }
};//代码最终错误，主要是边界问题处理不好啊啊啊啊啊
```

---

## 暴力解法

对于下标 i，下雨后水能到达的最大高度等于下标 i 两边的最大高度的最小值，下标 i 处能接的雨水量等于下标 i 处的水能到达的最大高度减去 height[i]。

朴素的做法是对于数组 height 中的每个元素，分别向左和向右扫描并记录左边和右边的最大高度，然后计算每个下标位置能接的雨水量。假设数组 height 的长度为 n，该做法需要对每个下标位置使用 O(n) 的时间向两边扫描并得到最大高度，因此总时间复杂度是 O(n2)。

---

## 方法一 动态规划！！！

动态规划（Dynamic Programming，简称 DP）是一种在数学、计算机科学和经济学等领域广泛应用的优化技术。它通过将复杂问题分解为相对简单的子问题，避免重复计算，从而提高求解效率。

**动态规划的核心思想：**

1. **最优子结构（Optimal Substructure）：** 问题的最优解可以通过其子问题的最优解组合而成。换句话说，问题的全局最优解包含了其子问题的全局最优解。
2. **子问题重叠（Overlapping Subproblems）：** 在解决问题的过程中，子问题会重复出现。动态规划通过记录已解决的子问题的解，避免重复计算。[维基百科，自由的百科全书+5博客园+5维基百科，自由的百科全书+5](https://www.cnblogs.com/Macw07/p/18590285?utm_source=chatgpt.com)

**动态规划的解决步骤：**

1. **定义状态：** 明确问题的状态表示，即用数学语言描述问题的各个阶段。通常使用数组或矩阵来存储这些状态。[维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/动态规划?utm_source=chatgpt.com)
2. **确定状态转移方程：** 建立状态之间的关系，明确如何通过已知状态推导出新的状态。这是动态规划的核心。[维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/动态规划?utm_source=chatgpt.com)
3. **设置初始条件：** 确定最初始状态的值，为后续状态的计算提供基础。
4. **计算并记录结果：** 按照状态转移方程，从初始状态开始，逐步计算并记录每个状态的最优解，直到求解出原问题的最优解。[博客园+3维基百科，自由的百科全书+3算法通关手册（LeetCode）+3](https://zh.wikipedia.org/wiki/动态规划?utm_source=chatgpt.com)

**动态规划的应用实例：**

- **背包问题：** 给定一组物品，每个物品有重量和价值，背包有最大承重，求在不超过背包重量的情况下，能够获得的最大价值。[维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/动态规划?utm_source=chatgpt.com)
- **硬币找零问题：** 给定不同面值的硬币和一个目标金额，求使用最少数量的硬币凑成该金额的方式。[维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/动态规划?utm_source=chatgpt.com)
- **最长公共子序列：** 给定两个序列，求它们的最长公共子序列的长度。[维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/动态规划?utm_source=chatgpt.com)

动态规划是一种强大的算法设计技术，能够有效解决具有最优子结构和子问题重叠性质的问题。掌握动态规划的思想和方法，对于解决复杂的优化问题具有重要意义。[维基百科，自由的百科全书+2维基百科，自由的百科全书+2](https://zh.wikipedia.org/wiki/动态规划?utm_source=chatgpt.com)

---



上述做法的时间复杂度较高是因为需要对每个下标位置都向两边扫描。如果已经知道每个位置两边的最大高度，则可以在 O(n) 的时间内得到能接的雨水总量。使用动态规划的方法，可以在 O(n) 的时间内预处理得到每个位置两边的最大高度。

创建两个长度为 n 的数组 leftMax 和 rightMax。对于 0≤i<n，leftMax[i] 表示下标 i 及其左边的位置中，height 的最大高度，rightMax[i] 表示下标 i 及其右边的位置中，height 的最大高度。

显然，leftMax[0]=height[0]，rightMax[n−1]=height[n−1]。两个数组的其余元素的计算如下：

    当 1≤i≤n−1 时，leftMax[i]=max(leftMax[i−1],height[i])；
    
    当 0≤i≤n−2 时，rightMax[i]=max(rightMax[i+1],height[i])。

因此可以正向遍历数组 height 得到数组 leftMax 的每个元素值，反向遍历数组 height 得到数组 rightMax 的每个元素值。

在得到数组 leftMax 和 rightMax 的每个元素值之后，对于 0≤i<n，下标 i 处能接的雨水量等于 min(leftMax[i],rightMax[i])−height[i]。遍历每个下标位置即可得到能接的雨水总量。

动态规划做法可以由下图体现。![fig1](https://assets.leetcode-cn.com/solution-static/42/1.png)

**注意重叠**

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        if (n == 0) {
            return 0;
        }
        vector<int> leftMax(n);
        leftMax[0] = height[0];
        for (int i = 1; i < n; ++i) {
            leftMax[i] = max(leftMax[i - 1], height[i]);//注意这里的自底向上思路，即状态转移方程
        }

        vector<int> rightMax(n);
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            rightMax[i] = max(rightMax[i + 1], height[i]);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans += min(leftMax[i], rightMax[i]) - height[i];//重叠的地方就是能接的水！！！
        }
        return ans;
    }
};
```

---

# 方法二 单调栈(之后细看)

除了计算并存储每个位置两边的最大高度以外，也可以用单调栈计算能接的雨水总量。

维护一个单调栈，单调栈存储的是下标，满足从栈底到栈顶的下标对应的数组 height 中的元素递减。

从左到右遍历数组，遍历到下标 i 时，如果栈内至少有两个元素，记栈顶元素为 top，top 的下面一个元素是 left，则一定有 height[left]≥height[top]。如果 height[i]>height[top]，则得到一个可以接雨水的区域，该区域的宽度是 i−left−1，高度是 min(height[left],height[i])−height[top]，根据宽度和高度即可计算得到该区域能接的雨水量。

为了得到 left，需要将 top 出栈。在对 top 计算能接的雨水量之后，left 变成新的 top，重复上述操作，直到栈变为空，或者栈顶下标对应的 height 中的元素大于或等于 height[i]。

在对下标 i 处计算能接的雨水量之后，将 i 入栈，继续遍历后面的下标，计算能接的雨水量。遍历结束之后即可得到能接的雨水总量。

下面用一个例子 height=[0,1,0,2,1,0,1,3,2,1,2,1] 来帮助读者理解单调栈的做法。

![img](https://assets.leetcode-cn.com/solution-static/42/f1.png)

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0;
        stack<int> stk;
        int n = height.size();
        for (int i = 0; i < n; ++i) {
            while (!stk.empty() && height[i] > height[stk.top()]) {//！！注意！！&&两边不能换顺序
                int top = stk.top();
                stk.pop();
                if (stk.empty()) {
                    break;
                }//排除左边不能积水的情况
                int left = stk.top();
                int currWidth = i - left - 1;//计算积水
                int currHeight = min(height[left], height[i]) - height[top];
                ans += currWidth * currHeight;
            }
            stk.push(i);
        }//高度递减的单调栈
        return ans;
    }
};
```

---

单调栈是一种栈（Stack）数据结构，其元素在栈内保持单调性，即栈内元素按照某种特定顺序（递增或递减）排列。这种特性使得单调栈在解决某些算法问题时非常高效，特别是在需要快速获取元素之间关系的问题中。

**单调栈的基本操作：**

1. **入栈（Push）：** 将新元素添加到栈顶。
2. **出栈（Pop）：** 移除栈顶元素。
3. **栈顶元素访问（Top）：** 获取栈顶元素的值，但不移除它。

**单调栈的应用场景：**

单调栈常用于解决需要比较相邻元素或寻找某些特定关系的问题，例如：

- **求解柱状图中接雨水的容量：** 在一维数组中，计算每个元素上方可以容纳的雨水量。
- **求解每日温度问题：** 给定一个数组，表示每日的温度，找出每一天距离下一次更高温度的天数。
- **求解股票价格的下一次更高价格：** 给定一个数组，表示股票的价格，找出每一天之后的更高价格的天数。

**单调栈的优势：**

- **时间效率：** 通过维护栈内元素的单调性，可以在遍历数组的过程中高效地解决问题，通常时间复杂度为 O(n)。
- **空间效率：** 使用栈来存储元素索引，避免了使用额外的数组或复杂的数据结构。

**示例：**

以解决“每日温度”问题为例，给定一个数组 `temperatures`，其中 `temperatures[i]` 代表第 `i` 天的温度。我们需要返回一个数组 `result`，其中 `result[i]` 表示第 `i` 天距离下一次更高温度的天数。

使用单调栈的思路如下：

1. **初始化：** 创建一个空栈 `stack`，用于存储温度的索引。

2. 遍历数组：

    从左到右遍历 

   ```
   temperatures
   ```

    数组。

   - 对于每一天的温度，检查栈顶元素对应的温度是否小于当前温度。
   - 如果是，表示找到了一个更高的温度，计算天数差，将栈顶元素出栈。
   - 将当前温度的索引入栈。

3. **结束：** 遍历结束后，栈中剩余的元素表示没有找到更高温度的日子，对应的结果设为 0。

**代码示例：**

```cpp
#include <vector>
using namespace std;

vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> result(n, 0);
    stack<int> s;
    for (int i = 0; i < n; ++i) {
        while (!s.empty() && temperatures[s.top()] < temperatures[i]) {
            int idx = s.top();
            s.pop();
            result[idx] = i - idx;
        }
        s.push(i);
    }
    return result;
}
```

在上述代码中，`s` 是一个栈，用于存储温度的索引。通过维护栈内元素的单调性（栈内温度递减），我们能够在一次遍历中高效地计算出每一天距离下一次更高温度的天数。

总而言之，单调栈是一种巧妙利用栈的特性来解决特定问题的技术，通过维护栈内元素的单调性，能够在许多场景下提供高效的解决方案。

---

# 方法三 双指针

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0;
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        while (left < right) {
            leftMax = max(leftMax, height[left]);
            rightMax = max(rightMax, height[right]);
            if (height[left] < height[right]) {//注意这里的判断条件
                ans += leftMax - height[left];
                ++left;
            } else {
                ans += rightMax - height[right];
                --right;
            }
        }
        return ans;
    }
};
```

**为什么要判断 `height[left] < height[right]`？**

这个判断用于决定移动哪个指针，以确保计算蓄水量时的正确性。

1. **当 `height[left] < height[right]` 时：**

   - **原因：** 如果左边的高度较小，意味着左边的墙壁限制了水的蓄积高度。
   - **操作：** 计算当前位置的蓄水量，将其累加到 `ans` 中，然后移动 `left` 指针向右移动一格，尝试找到更高的左边墙壁。

2. **当 `height[left] >= height[right]` 时：**

   - **原因：** 如果右边的高度较小，意味着右边的墙壁限制了水的蓄积高度。

   - **操作：** 计算当前位置的蓄水量，将其累加到 `ans` 中，然后移动 `right` 指针向左移动一格，尝试找到更高的右边墙壁。

     

动态规划的做法中，需要维护两个数组 leftMax 和 rightMax，因此空间复杂度是 O(n)。是否可以将空间复杂度降到 O(1)？

注意到下标 i 处能接的雨水量由 leftMax[i] 和 rightMax[i] 中的最小值决定。由于数组 leftMax 是从左往右计算，数组 rightMax 是从右往左计算，因此可以使用双指针和两个变量代替两个数组。

维护两个指针 left 和 right，以及两个变量 leftMax 和 rightMax，初始时 left=0,right=n−1,leftMax=0,rightMax=0。指针 left 只会向右移动，指针 right 只会向左移动，在移动指针的过程中维护两个变量 leftMax 和 rightMax 的值。

当两个指针没有相遇时，进行如下操作：

    使用 height[left] 和 height[right] 的值更新 leftMax 和 rightMax 的值；
    
    如果 height[left]<height[right]，则必有 leftMax<rightMax，下标 left 处能接的雨水量等于 leftMax−height[left]，将下标 left 处能接的雨水量加到能接的雨水总量，然后将 left 加 1（即向右移动一位）；
    
    如果 height[left]≥height[right]，则必有 leftMax≥rightMax，下标 right 处能接的雨水量等于 rightMax−height[right]，将下标 right 处能接的雨水量加到能接的雨水总量，然后将 right 减 1（即向左移动一位）。

当两个指针相遇时，即可得到能接的雨水总量。
