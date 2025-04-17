# leetcode8

```cpp
std::string p;
for (const char& iter : p)
{
    // 处理 iter
}
```

是合法的 C++ 语法。这段代码使用范围基于范围的 for 循环（range-based for loop）遍历字符串 `p` 中的每个字符。在循环中，`iter` 是对字符串中每个字符的常量引用，您可以在循环体内使用它来处理每个字符。

需要注意的是，如果您在循环中需要修改字符串的字符，`iter` 应该声明为非常量引用，即 `char& iter`。然而，直接修改字符串的字符通常不推荐，因为这可能会影响字符串的完整性。

此外，`std::string` 提供了多种成员函数来操作字符串，例如 `c_str()`、`data()` 和 `copy()` 等。其中，`c_str()` 返回一个指向以空字符终止的 C 风格字符串的指针，适用于需要 C 风格字符串的场景。`data()` 返回一个指向非空字符终止的字符数组的指针，适用于需要原始字符数组的场景。`copy()` 函数用于将字符串的内容复制到另一个字符数组中。 [菜鸟教程+8CSDN博客+8byvoid.com+8](https://blog.csdn.net/u013066730/article/details/88893728?utm_source=chatgpt.com)

总之，您的代码在语法上是正确的，使用范围基于 for 循环遍历 `std::string` 中的字符是一种常见且有效的做法。

---

push_back跟insert分别用在哪些地方：

在 C++ 中，`push_back` 和 `insert` 都用于向容器中添加元素，但它们的使用场景和行为有所不同。

**`push_back`**：

- **功能**：将一个元素添加到容器的末尾。[CSDN博客+2知乎专栏+2知乎专栏+2](https://zhuanlan.zhihu.com/p/682517443?utm_source=chatgpt.com)
- **适用场景**：当您需要在容器末尾添加单个元素时，使用 `push_back` 是最直接和高效的选择。
- **示例**：

```cpp
  std::vector<int> numbers;
  numbers.push_back(10); // 在末尾添加元素 10
```

**`insert`**：

- **功能**：在容器的指定位置插入一个或多个元素。[CSDN博客](https://blog.csdn.net/weixin_43632918/article/details/109901164?utm_source=chatgpt.com)
- **适用场景**：当您需要在容器的特定位置（而非末尾）插入元素时，使用 `insert`。
- **示例**：

```cpp
  std::vector<int> numbers = {1, 2, 4, 5};
  auto it = numbers.begin() + 2; // 指向第三个元素
  numbers.insert(it, 3); // 在第三个位置插入元素 3
```

**性能考虑**：

- **`push_back`**：在 `std::vector` 等动态数组类型的容器中，`push_back` 可能导致重新分配内存，特别是在容器容量不足以容纳新元素时。这是因为需要分配更大的内存空间，将现有元素复制到新位置，然后添加新元素。为了减少这种开销，可以在预知需要添加大量元素时，使用 `reserve` 方法预先分配足够的容量。
- **`insert`**：在 `std::vector` 等容器中，`insert` 会将指定位置及其后面的所有元素向后移动，以腾出空间插入新元素。这可能导致较高的时间复杂度，特别是在容器中间插入大量元素时。对于 `std::list` 等双向链表类型的容器，`insert` 操作的性能影响较小，因为它们不需要移动元素。

**总结**：

- 使用 `push_back` 在容器末尾添加元素，适用于大多数需要在末尾添加单个元素的场景。[CSDN博客+2知乎专栏+2知乎专栏+2](https://zhuanlan.zhihu.com/p/213853588?utm_source=chatgpt.com)
- 使用 `insert` 在容器的指定位置插入元素，适用于需要在特定位置添加元素的场景。
- 在使用 `insert` 时，特别是在容器中间插入多个元素时，需要注意性能开销。

根据具体的需求和容器类型，选择适当的插入方法，以确保代码的效率和可读性。

---

那再跟emplace比较呢：

在 C++ 中，`std::set` 提供了多种方法来向集合中添加元素，其中 `insert`、`emplace` 和 `emplace_hint` 是常用的三种方式。它们各有特点，适用于不同的场景。

**`insert` 方法**：

- **功能**：将一个已存在的元素添加到集合中。
- **使用方式**：需要先创建一个元素对象，然后将其传递给 `insert`。例如：

```cpp
 std::set<int> numbers;
  int value = 10;
  numbers.insert(value); // 将 value 插入集合
```

在此过程中，`insert` 会将 `value` 插入到集合中。如果集合中已存在相同的元素，插入操作不会生效。

**`emplace` 方法**：

- **功能**：在集合中直接原地构造元素，避免了不必要的拷贝或移动操作。[Wikipedia+2维基百科，自由的百科全书+2维基百科，自由的百科全书+2](https://zh.wikipedia.org/wiki/关联性容器?utm_source=chatgpt.com)
- **使用方式**：直接传递构造元素所需的参数，集合会在内部使用这些参数直接构造元素。例如：

```cpp
  std::set<std::pair<int, std::string>> mySet;
  mySet.emplace(1, "apple"); // 直接在集合中构造 std::pair<int, std::string>(1, "apple")
```

相比于 `insert`，`emplace` 通过原地构造，避免了临时对象的创建和拷贝，提高了性能。

**`emplace_hint` 方法**：

- **功能**：在集合中直接原地构造元素，并提供一个位置提示，以优化插入操作的效率。[维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/关联性容器?utm_source=chatgpt.com)
- **使用方式**：需要提供一个迭代器作为提示位置，以及构造元素所需的参数。例如：

```cpp
  std::set<int> numbers = {1, 3, 5};
  auto it = numbers.begin();
  std::advance(it, 1); // 将迭代器移动到第二个位置
  numbers.emplace_hint(it, 4); // 在第二个位置之前插入元素 4
```

`emplace_hint` 在提供了插入位置提示的情况下，可以比 `emplace` 更高效地执行插入操作。[C语言中文网+1GeeksforGeeks+1](https://c.biancheng.net/view/7197.html?utm_source=chatgpt.com)

**总结**：

- **`insert`**：适用于将已存在的元素添加到集合中，但可能涉及拷贝或移动操作。
- **`emplace`**：适用于直接在集合中构造新元素，避免了不必要的拷贝或移动操作，通常性能更优。
- **`emplace_hint`**：在提供插入位置提示的情况下，原地构造元素，进一步优化插入性能。

在选择使用哪种方法时，应考虑元素的构造成本、是否需要提供插入位置提示以及对性能的要求。

---

在同等情况下，相比于.find()，.count()更高效

---

```cpp
//自己的代码，有错误，abc为p时abb也判断为正确
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
         unordered_set<char> charSet;
         vector<int> result;
         for(const char& iter : p){
            charSet.emplace(iter);
         }
         int slowIndex = 0;
         for(int fastIndex = 0; fastIndex < s.size(); fastIndex++){
            if(charSet.count(s[fastIndex]) == 0){
                fastIndex++;
                slowIndex = fastIndex;
                continue;
            }
            if(fastIndex - slowIndex == p.size()){
                slowIndex = fastIndex;
                continue;
            }
            if(fastIndex - slowIndex == p.size() - 1){
                result.emplace_back(slowIndex++);
            }
         }
         return result;
    }
};
```

---

## 方法一 滑动窗口

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int sLen = s.size(), pLen = p.size();

        if (sLen < pLen) {
            return vector<int>();
        }

        vector<int> ans;
        vector<int> sCount(26);
        vector<int> pCount(26);//这里用数组好判断字母重复的情况
        for (int i = 0; i < pLen; ++i) {
            ++sCount[s[i] - 'a'];
            ++pCount[p[i] - 'a'];
        }

        if (sCount == pCount) {
            ans.emplace_back(0);
        }

        for (int i = 0; i < sLen - pLen; ++i) {
            --sCount[s[i] - 'a'];
            ++sCount[s[i + pLen] - 'a'];

            if (sCount == pCount) {//逆天但精妙，直接看俩数组相不相同
                ans.emplace_back(i + 1);
            }
        }

        return ans;
    }
};
```

---

## 方法二 优化滑动窗口

方法一的基础上，我们不再分别统计滑动窗口和字符串 p 中每种字母的数量，而是统计滑动窗口和字符串 p 中每种字母数量的差；并引入变量 differ 来记录当前窗口与字符串 p 中数量不同的字母的个数，并在滑动窗口的过程中维护它。

在判断滑动窗口中每种字母的数量与字符串 p 中每种字母的数量是否相同时，只需要判断 differ 是否为零即可。

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int sLen = s.size(), pLen = p.size();

        if (sLen < pLen) {
            return vector<int>();
        }

        vector<int> ans;
        vector<int> count(26);
        for (int i = 0; i < pLen; ++i) {
            ++count[s[i] - 'a'];
            --count[p[i] - 'a'];
        }

        int differ = 0;
        for (int j = 0; j < 26; ++j) {
            if (count[j] != 0) {
                ++differ;
            }
        }

        if (differ == 0) {
            ans.emplace_back(0);
        }

        for (int i = 0; i < sLen - pLen; ++i) {
            if (count[s[i] - 'a'] == 1) {  // 窗口中字母 s[i] 的数量与字符串 p 中的数量从不同变得相同
                --differ;
            } else if (count[s[i] - 'a'] == 0) {  // 窗口中字母 s[i] 的数量与字符串 p 中的数量从相同变得不同
                ++differ;
            }
            --count[s[i] - 'a'];

            if (count[s[i + pLen] - 'a'] == -1) {  // 窗口中字母 s[i+pLen] 的数量与字符串 p 中的数量从不同变得相同
                --differ;
            } else if (count[s[i + pLen] - 'a'] == 0) {  // 窗口中字母 s[i+pLen] 的数量与字符串 p 中的数量从相同变得不同
                ++differ;
            }
            ++count[s[i + pLen] - 'a'];
            
            if (differ == 0) {
                ans.emplace_back(i + 1);
            }
        }

        return ans;
    }
};

```

