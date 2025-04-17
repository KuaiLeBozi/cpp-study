# leetcode1

#### 暴力枚举

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();//避免重复访问！！！
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return {i, j};
                }
            }
        }
        return {};
    }
};
```

return {}为**返回一个临时vector数组索引**，很方便:

- `return {i, j};` 语法创建一个临时的 [vector](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)，包含 [i](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) 和 [j](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) 两个元素，并将其返回。
- 如果没有找到满足条件的两个数，函数返回一个空的 [vector](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)。

class:

- `class` 关键字用于定义一个类。
- [Solution](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) 是类的名称。
- 类是C++中的一种用户定义的数据类型，它可以包含数据成员（变量）和成员函数（方法）。

public：

- `public` 是一个访问控制修饰符。
- 在 `public` 访问控制下定义的成员可以被类的外部访问。
- 其他访问控制修饰符包括 `private` 和 `protected`，它们限制了成员的访问权限。

---

#### 哈希表

```CPP
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashtable;
        for (int i = 0; i < nums.size(); ++i) {
            auto it = hashtable.find(target - nums[i]);//在哈希表中寻找是否存在target-nums[i]
            if (it != hashtable.end()) {//如果没有找到，find 方法返回一个指向哈希表末尾的迭代器
                return {it->second, i};//map相当于数据结构,second为索引,first为值
            }
            hashtable[nums[i]] = i;//前面是值，后面是索引
        }
        return {};
    }
};
```



- [unordered_map](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) 定义了一个键为整数、值为整数的哈希表。
- 键（`int`）表示数组中的数，值（`int`）表示该数在数组中的索引。

总结：以上哈希表并未先把nums里的元素存进hashtable，而是==遍历一个存一个==，直到找到存的是元素。**注意看看hashtable是怎么用的**！！！

