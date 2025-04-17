# leetcode2

```cpp
//初始自己写的代码
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> result;
        vector<string> s=strs;
        for(int m=0;m<strs.size();m++)
            sort(s[m].begin(),s[m].end());//先将里面的字符串元素进行排序，方便比较
        for(int i=0;i<strs.size();i++){
            vector<string> temp={strs[i]};
            for(int j=i+1;j<strs.size();j++){
                if(s[i]==s[j]) temp.push_back(strs[j]);
            }
            result.push_back(temp);
        }
        return result;
    }
};
```

输出结果

```cpp
[["eat","tea","ate"],["tea","ate"],["tan","nat"],["ate"],["nat"],["bat"]]
```

发现有重复的情况，这种情况也只能想到用==哈希表==进行解决

---

#### 方法一：排序

由于互为字母异位词的两个字符串包含的字母相同，因此对两个字符串**分别进行排序**之后得到的字符串一定是相同的，故可以将排序之后的字符串作为==哈希表的键==

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;//前键后值
        for (string& str: strs) {//这一段代码是C++ 经典解法 里处理 字母异位词分组（group 										//anagrams） 的核心循环,&为引用符，避免拷贝
            string key = str;
            sort(key.begin(), key.end());
            mp[key].emplace_back(str);//emplace_back(str) 是 push_back(str) 的增强版，直接在 									//vector 里原地构造 str，且这里也有没有元素便创建的功能
        }
        vector<vector<string>> ans;
        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.emplace_back(it->second);//此时每个键底下已经有对应键的单词了，且均为一个字符串数组
        }
        return ans;
    }
};
```

* c语言里面的&是“取地址符”，是用来**取变量的内存地址**，**没有"引用"这一说法**

```c
int a = 10;
int *p = &a;  // &a 表示“取 a 的地址”
```

* c++的引用就是给某个变量**起个别名**，直接**操作这个别名就相当于操作原变量**

```cpp
int a = 10;
int& ref = a;  // ref 是 a 的引用（别名）
ref = 20;      // 相当于 a = 20;
cout << a;     // 输出 20
```

* c++也可做"取地址符"

```cpp
int a = 10;
int *p = &a;  // 取地址
```

------

## ✅ `emplace_back(str)` 是什么？

### 📌 定义：

`emplace_back()` 是 `C++11` 引入的 `vector` 成员函数，用来 **直接在 `vector` 内部构造元素**，避免不必要的拷贝或移动。

==注意是构造元素==

### 📌 作用：

- **在 `vector` 末尾** **原地** 构造一个元素
- 参数直接传给元素的构造函数
- 比 `push_back()` 更高效（某些场景下）

---

```cpp
for (string& str : strs)
```

## ✅ 直接翻译：

意思是：

- 遍历 `strs`（这是一个 `vector<string>`）
- 每次循环，把当前元素的 **引用** 赋给 `str`

### **核心重点：**

- `string& str` 里的 `&` 表示 **引用**
- `str` 是 `strs` 里某个字符串的 **别名**
- 修改 `str`，就是直接修改 `strs` 里的那个元素本身

## ✅ 和普通 `for (string str : strs)` 的区别：

| 写法                       | 说明                                                | 是否拷贝？ |
| -------------------------- | --------------------------------------------------- | ---------- |
| `for (string str : strs)`  | 每次循环，`str` 都是 `strs` 里元素的 **拷贝**       | ✅ 会拷贝   |
| `for (string& str : strs)` | 每次循环，`str` 是 `strs` 里元素的 **引用**（别名） | ❌ 不拷贝   |

---

#### 方法二：计数(比较难，晚点再看)

由于互为字母异位词的两个字符串包含的字母相同，因此两个字符串中的相同字母出现的次数一定是相同的，故可以将每个字母出现的次数使用字符串表示，作为哈希表的键。

由于字符串只包含小写字母，因此对于每个字符串，可以使用长度为 26 的数组记录每个字母出现的次数。需要注意的是，在使用数组作为哈希表的键时，不同语言的支持程度不同，因此不同语言的实现方式也不同。

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        // 自定义对 array<int, 26> 类型的哈希函数
        auto arrayHash = [fn = hash<int>{}] (const array<int, 26>& arr) -> size_t {
            return accumulate(arr.begin(), arr.end(), 0u, [&](size_t acc, int num) {
                return (acc << 1) ^ fn(num);
            });
        };

        unordered_map<array<int, 26>, vector<string>, decltype(arrayHash)> mp(0, arrayHash);
        for (string& str: strs) {
            array<int, 26> counts{};
            int length = str.length();
            for (int i = 0; i < length; ++i) {
                counts[str[i] - 'a'] ++;
            }
            mp[counts].emplace_back(str);
        }
        vector<vector<string>> ans;
        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.emplace_back(it->second);
        }
        return ans;
    }
};
```
复杂度分析

* 时间复杂度：O(n(k+∣Σ∣))，其中 n 是 strs 中的字符串的数量，k 是 strs 中的字符串的的最大长度，Σ 是字符集，在本题中字符集为所有小写字母，∣Σ∣=26。需要遍历 n 个字符串，对于每个字符串，需要 O(k) 的时间计算每个字母出现的次数，O(∣Σ∣) 的时间生成哈希表的键，以及 O(1) 的时间更新哈希表，因此总时间复杂度是 O(n(k+∣Σ∣))。
* 空间复杂度：O(n(k+∣Σ∣))，其中 n 是 strs 中的字符串的数量，k 是 strs 中的字符串的最大长度，Σ 是字符集，在本题中字符集为所有小写字母，∣Σ∣=26。需要用哈希表存储全部字符串，而记录每个字符串中每个字母出现次数的数组需要的空间为 O(∣Σ∣)，在渐进意义下小于 O(n(k+∣Σ∣))，可以忽略不计。



