"""
========================================================
LeetCode Problems Collection
========================================================
Includes:
1. Find Pivot Index
2. Largest Number At Least Twice of Others
3. Plus One
4. Diagonal Traverse
5. Spiral Matrix
6. Pascal's Triangle
7. Add Binary
8. Longest Common Prefix
9. Implement strStr()
10. Reverse_string
11. remove_element
12. two_sum_sorted
13. find_max_consecutive_ones
14. min_subarray_len
15. pascal_triangle_ii
16. reverse_words
17. reverse_words_iii
18. remove_duplicates
19. move_zeroes
20. (Placeholder for future problem)

Includes:
- Problem descriptions
- Solutions
- Unit tests
- main() runner
========================================================
"""

import unittest


# ======================================================
# Problem 1: Find Pivot Index
# ======================================================
# Given an array of integers nums, calculate the pivot index of this array.
#
# The pivot index is the index where the sum of all the numbers strictly
# to the left of the index is equal to the sum of all the numbers strictly
# to the right.
#
# If the index is on the left edge of the array, then the left sum is 0.
# If the index is on the right edge of the array, then the right sum is 0.
#
# Return the leftmost pivot index. If no such index exists, return -1.
#
# Examples:
# Input: nums = [1,7,3,6,5,6]
# Output: 3
#
# Input: nums = [1,2,3]
# Output: -1
#
# Input: nums = [2,1,-1]
# Output: 0
#
# Constraints:
# 1 <= nums.length <= 10^4
# -1000 <= nums[i] <= 1000
#
# Note:
# This problem is the same as LeetCode 1991.
# https://leetcode.com/problems/find-the-middle-index-in-array/
# ======================================================

def pivot_index(nums: list[int]) -> int:
    return 


# ======================================================
# Problem 2: Largest Number At Least Twice of Others
# ======================================================
# You are given an integer array nums where the largest integer is unique.
#
# Determine whether the largest element in the array is at least twice
# as much as every other number in the array.
#
# If it is, return the index of the largest element.
# Otherwise, return -1.
#
# Examples:
# Input: nums = [3,6,1,0]
# Output: 1
#
# Input: nums = [1,2,3,4]
# Output: -1

# Constraints:
# 2 <= nums.length <= 50
# 0 <= nums[i] <= 100
# The largest element in nums is unique.
# ======================================================

def dominant_index(nums: list[int]) -> int:
    max_value = max(nums)
    max_index = nums.index(max_value)

    for i, num in enumerate(nums):
        if i != max_index and max_value < 2 * num:
            return -1

    return max_index


# ======================================================
# Problem 3: Plus One
# ======================================================
# You are given a large integer represented as an integer array digits.
#
# Each digits[i] is the ith digit of the integer.
# The digits are ordered from most significant to least significant.
#
# Increment the large integer by one and return the resulting array.
#
# Examples:
# Input: digits = [1,2,3]
# Output: [1,2,4]
#
# Input: digits = [4,3,2,1]
# Output: [4,3,2,2]
#
# Input: digits = [9]
# Output: [1,0]
#
# Constraints:
# 1 <= digits.length <= 100
# 0 <= digits[i] <= 9
# digits does not contain any leading 0's.
# ======================================================

def plus_one(digits: list[int]) -> list[int]:
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0

    return [1] + digits


# ======================================================
# Problem 4: Diagonal Traverse
# ======================================================
# Given an m x n matrix, return all elements in diagonal order.
#
# Example:
# Input: [[1,2,3],[4,5,6],[7,8,9]]
# Output: [1,2,4,7,5,3,6,8,9]
#
# Constraints:
# 1 <= mat.length, mat[i].length <= 10^4
# -10^5 <= mat[i][j] <= 10^5
# ======================================================

def find_diagonal_order(mat: list[list[int]]) -> list[int]:
    if not mat or not mat[0]:
        return []

    m, n = len(mat), len(mat[0])
    result = []

    for d in range(m + n - 1):
        intermediate = []

        row = 0 if d < n else d - n + 1
        col = d if d < n else n - 1

        while row < m and col >= 0:
            intermediate.append(mat[row][col])
            row += 1
            col -= 1

        if d % 2 == 0:
            result.extend(intermediate[::-1])
        else:
            result.extend(intermediate)

    return result


# ======================================================
# Problem 5: Spiral Matrix
# ======================================================
# Given an m x n matrix, return all elements of the matrix in spiral order.
#
# Example:
# Input: [[1,2,3],[4,5,6],[7,8,9]]
# Output: [1,2,3,6,9,8,7,4,5]
#
# Constraints:
# 1 <= m, n <= 10
# -100 <= matrix[i][j] <= 100
# ======================================================

def spiral_order(matrix: list[list[int]]) -> list[int]:
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result


# ======================================================
# Problem 6: Pascal's Triangle
# ======================================================
# Given an integer numRows, return the first numRows of Pascal's triangle.
#
# In Pascal's triangle, each number is the sum of the two numbers
# directly above it as shown:
#
#     1
#    1 1
#   1 2 1
#  1 3 3 1
# 1 4 6 4 1
#
# Example:
# Input: numRows = 5
# Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
#
# Constraints:
# 1 <= numRows <= 30
# ======================================================

def generate_pascals_triangle(numRows: int) -> list[list[int]]:
    triangle = []

    for i in range(numRows):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        triangle.append(row)

    return triangle


# ======================================================
# Problem 7: Add Binary
# ======================================================
# Given two binary strings a and b, return their sum as a binary string.
#
# Example:
# Input: a = "11", b = "1"
# Output: "100"
#
# Input: a = "1010", b = "1011"
# Output: "10101"
#
# Constraints:
# 1 <= a.length, b.length <= 10^4
# a and b consist only of '0' or '1' characters.
# Each string does not contain leading zeros except for the zero itself.
# ======================================================

def add_binary(a: str, b: str) -> str:
    i, j = len(a) - 1, len(b) - 1
    carry = 0
    result = []

    while i >= 0 or j >= 0 or carry:
        total = carry
        if i >= 0:
            total += int(a[i])
            i -= 1
        if j >= 0:
            total += int(b[j])
            j -= 1

        result.append(str(total % 2))
        carry = total // 2

    return "".join(reversed(result))


# ======================================================
# Problem 8: Longest Common Prefix
# ======================================================
# Write a function to find the longest common prefix string
# amongst an array of strings.
#
# If there is no common prefix, return an empty string "".
#
# Example:
# Input: strs = ["flower","flow","flight"]
# Output: "fl"
#
# Input: strs = ["dog","racecar","car"]
# Output: ""
# Explanation: There is no common prefix among the input strings.
#
# Constraints:
# 1 <= strs.length <= 200
# 0 <= strs[i].length <= 200
# strs[i] consists of only lowercase English letters.
# ======================================================

def longest_common_prefix(strs: list[str]) -> str:
    if not strs:
        return ""

    prefix = strs[0]

    for word in strs[1:]:
        while not word.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""

    return prefix


# ======================================================
# Problem 9: Implement strStr()
# ======================================================
# Return the index of the first occurrence of needle in haystack,
# or -1 if needle is not part of haystack.
#
# Clarification:
# When needle is an empty string, we should return 0.
# This matches C's strstr() and Java's indexOf().
#
# Example:
# Input: haystack = "hello", needle = "ll"
# Output: 2
#
# Input: haystack = "aaaaa", needle = "bba"
# Output: -1
#
# Input: haystack = "sadbutsad", needle = "sad"
# Output: 0
#
# Constraints:
# 0 <= haystack.length, needle.length <= 5 * 10^4
# haystack and needle consist of only lowercase English characters.
# ======================================================

def str_str(haystack: str, needle: str) -> int:
    if needle == "":
        return 0

    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i

    return -1


# ======================================================
# Problem 10: Reverse String (Two Pointers)
# ======================================================
# Write a function that reverses a string.
# The input string is given as an array of characters s.
#
# You must do this by modifying the input array in-place
# with O(1) extra memory.
#
# Examples:
# Input: s = ["h","e","l","l","o"]
# Output: ["o","l","l","e","h"]
#
# Input: s = ["H","a","n","n","a","h"]
# Output: ["h","a","n","n","a","H"]
#
# Constraints:
# 1 <= s.length <= 10^5
# s[i] is a printable ASCII character.
# ======================================================

def reverse_string(s: list[str]) -> None:
    left, right = 0, len(s) - 1

    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


# ======================================================
# Problem 11: Remove Element (Two Pointers)
# ======================================================
# Given an integer array nums and an integer val,
# remove all occurrences of val in nums in-place.
#
# Return the number of elements k that are not equal to val.
# The first k elements of nums should contain the result.
#
# Examples:
# Input: nums = [3,2,2,3], val = 3
# Output: 2, nums = [2,2,_,_]
#
# Input: nums = [0,1,2,2,3,0,4,2], val = 2
# Output: 5, nums = [0,1,4,0,3,_,_,_]
#
# Constraints:
# 0 <= nums.length <= 100
# 0 <= nums[i] <= 50
# 0 <= val <= 100
# ======================================================

def remove_element(nums: list[int], val: int) -> int:
    k = 0

    for num in nums:
        if num != val:
            nums[k] = num
            k += 1

    return k


# ======================================================
# Problem 12: Two Sum II - Input Array Is Sorted
# ======================================================
# Given a 1-indexed array of integers numbers that is
# already sorted in non-decreasing order, find two numbers
# such that they add up to a specific target.
#
# Return the indices of the two numbers (1-based).
# Exactly one solution exists.
#
# Examples:
# Input: numbers = [2,7,11,15], target = 9
# Output: [1,2]
#
# Input: numbers = [2,3,4], target = 6
# Output: [1,3]
#
# Constraints:
# 2 <= numbers.length <= 3 * 10^4
# -1000 <= numbers[i] <= 1000
# numbers is sorted in non-decreasing order
# -1000 <= target <= 1000
# ======================================================

def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1


# ======================================================
# Problem 13: Max Consecutive Ones
# ======================================================
# Given a binary array nums, return the maximum number
# of consecutive 1s in the array.
#
# Examples:
# Input: nums = [1,1,0,1,1,1]
# Output: 3
#
# Input: nums = [1,0,1,1,0,1]
# Output: 2
#
# Constraints:
# 1 <= nums.length <= 10^5
# nums[i] is either 0 or 1
# ======================================================

def find_max_consecutive_ones(nums: list[int]) -> int:
    max_count = 0
    current = 0

    for num in nums:
        if num == 1:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0

    return max_count


# ======================================================
# Problem 14: Minimum Size Subarray Sum (Sliding Window)
# ======================================================
# Given an array of positive integers nums and a positive
# integer target, return the minimal length of a subarray
# whose sum is greater than or equal to target.
#
# If no such subarray exists, return 0.
#
# Examples:
# Input: target = 7, nums = [2,3,1,2,4,3]
# Output: 2
#
# Input: target = 4, nums = [1,4,4]
# Output: 1
#
# Input: target = 11, nums = [1,1,1,1,1,1,1,1]
# Output: 0
#
# Constraints:
# 1 <= target <= 10^9
# 1 <= nums.length <= 10^5
# 1 <= nums[i] <= 10^4
# ======================================================

def min_subarray_len(target: int, nums: list[int]) -> int:
    left = 0
    total = 0
    min_length = float('inf')

    for right in range(len(nums)):
        total += nums[right]

        while total >= target:
            min_length = min(min_length, right - left + 1)
            total -= nums[left]
            left += 1

    return 0 if min_length == float('inf') else min_length


# ======================================================
# Problem 15: Pascal's Triangle II
# ======================================================
# Given an integer rowIndex, return the rowIndexth (0-indexed)
# row of the Pascal's triangle.
#
# In Pascal's triangle, each number is the sum of the two numbers
# directly above it as shown:
#
#     1
#    1 1
#   1 2 1
#  1 3 3 1
# 1 4 6 4 1
#
# Example 1:
# Input: rowIndex = 3
# Output: [1,3,3,1]
#
# Example 2:
# Input: rowIndex = 0
# Output: [1]
#
# Example 3:
# Input: rowIndex = 1
# Output: [1,1]
#
# Constraints:
# 0 <= rowIndex <= 33
# ======================================================

def get_pascal_row(rowIndex: int) -> list[int]:
    row = [1] * (rowIndex + 1)
    
    for i in range(1, rowIndex):
        # Compute from right to left to avoid overwriting values
        for j in range(i, 0, -1):
            row[j] = row[j] + row[j - 1]
    
    return row


# ======================================================
# Problem 16: Reverse Words in a String
# ======================================================
# Given an input string s, reverse the order of the words.
#
# A word is defined as a sequence of non-space characters.
# The words in s will be separated by at least one space.
# Return a string of the words in reverse order concatenated
# by a single space.
#
# Note that s may contain leading or trailing spaces or
# multiple spaces between two words. The returned string
# should only have a single space separating the words.
#
# Example 1:
# Input: s = "the sky is blue"
# Output: "blue is sky the"
#
# Example 2:
# Input: s = "  hello world  "
# Output: "world hello"
#
# Example 3:
# Input: s = "a good   example"
# Output: "example good a"
#
# Constraints:
# 1 <= s.length <= 10^4
# s contains English letters, digits, and spaces
# There is at least one word in s.
# ======================================================

def reverse_words(s: str) -> str:
    # Split the string by spaces and filter out empty strings
    words = [word for word in s.split() if word]
    # Reverse the list and join with single space
    return ' '.join(reversed(words))


# ======================================================
# Problem 17: Reverse Words in a String III
# ======================================================
# Given a string s, reverse the order of characters in each
# word within a sentence while still preserving whitespace
# and initial word order.
#
# Example 1:
# Input: s = "Let's take LeetCode contest"
# Output: "s'teL ekat edoCteeL tsetnoc"
#
# Example 2:
# Input: s = "Mr Ding"
# Output: "rM gniD"
#
# Constraints:
# 1 <= s.length <= 5 * 10^4
# s contains printable ASCII characters.
# s does not contain any leading or trailing spaces.
# There is at least one word in s.
# All the words in s are separated by a single space.
# ======================================================

def reverse_words_iii(s: str) -> str:
    # Split the string into words
    words = s.split()
    # Reverse each word
    reversed_words = [word[::-1] for word in words]
    # Join them back with single space
    return ' '.join(reversed_words)


# ======================================================
# Problem 18: Remove Duplicates from Sorted Array
# ======================================================
# Given an integer array nums sorted in non-decreasing order,
# remove the duplicates in-place such that each unique element
# appears only once.
#
# Return the number of unique elements k.
# The first k elements of nums should contain the unique numbers
# in sorted order.
#
# Example 1:
# Input: nums = [1,1,2]
# Output: 2, nums = [1,2,_]
#
# Example 2:
# Input: nums = [0,0,1,1,1,2,2,3,3,4]
# Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
#
# Constraints:
# 1 <= nums.length <= 3 * 10^4
# -100 <= nums[i] <= 100
# nums is sorted in non-decreasing order.
# ======================================================

def remove_duplicates(nums: list[int]) -> int:
    if not nums:
        return 0
    
    k = 1  # First element is always unique
    
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[k] = nums[i]
            k += 1
    
    return k


# ======================================================
# Problem 19: Move Zeroes
# ======================================================
# Given an integer array nums, move all 0's to the end of it
# while maintaining the relative order of the non-zero elements.
#
# Note that you must do this in-place without making a copy
# of the array.
#
# Example 1:
# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]
#
# Example 2:
# Input: nums = [0]
# Output: [0]
#
# Constraints:
# 1 <= nums.length <= 10^4
# -2^31 <= nums[i] <= 2^31 - 1
# ======================================================

def move_zeroes(nums: list[int]) -> None:
    # Position where the next non-zero element should go
    non_zero_pos = 0
    
    # Move all non-zero elements to the front
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[non_zero_pos] = nums[i]
            non_zero_pos += 1
    
    # Fill the remaining positions with zeros
    for i in range(non_zero_pos, len(nums)):
        nums[i] = 0


# ======================================================
# Unit Tests
# ======================================================

class TestLeetCodeProblems(unittest.TestCase):

    def test_pivot_index(self):
        self.assertEqual(pivot_index([1,7,3,6,5,6]), 3)
        self.assertEqual(pivot_index([1,2,3]), -1)
        self.assertEqual(pivot_index([2,1,-1]), 0)

    def test_dominant_index(self):
        self.assertEqual(dominant_index([3,6,1,0]), 1)
        self.assertEqual(dominant_index([1,2,3,4]), -1)

    def test_plus_one(self):
        self.assertEqual(plus_one([1,2,3]), [1,2,4])
        self.assertEqual(plus_one([4,3,2,1]), [4,3,2,2])
        self.assertEqual(plus_one([9]), [1,0])
        self.assertEqual(plus_one([9,9,9]), [1,0,0,0])

    def test_diagonal_traverse(self):
        self.assertEqual(
            find_diagonal_order([[1,2,3],[4,5,6],[7,8,9]]),
            [1,2,4,7,5,3,6,8,9]
        )

    def test_spiral_matrix(self):
        self.assertEqual(
            spiral_order([[1,2,3],[4,5,6],[7,8,9]]),
            [1,2,3,6,9,8,7,4,5]
        )

    def test_pascals_triangle(self):
        self.assertEqual(
            generate_pascals_triangle(5),
            [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
        )

    def test_add_binary(self):
        self.assertEqual(add_binary("11", "1"), "100")
        self.assertEqual(add_binary("1010", "1011"), "10101")

    def test_longest_common_prefix(self):
        self.assertEqual(
            longest_common_prefix(["flower","flow","flight"]),
            "fl"
        )
        self.assertEqual(
            longest_common_prefix(["dog","racecar","car"]),
            ""
        )

    def test_str_str(self):
        self.assertEqual(str_str("sadbutsad", "sad"), 0)
        self.assertEqual(str_str("leetcode", "leeto"), -1)
        self.assertEqual(str_str("hello", "ll"), 2)
        self.assertEqual(str_str("aaaaa", "bba"), -1)

    def test_reverse_string(self):
        s1 = ["h","e","l","l","o"]
        reverse_string(s1)
        self.assertEqual(s1, ["o","l","l","e","h"])
        
        s2 = ["H","a","n","n","a","h"]
        reverse_string(s2)
        self.assertEqual(s2, ["h","a","n","n","a","H"])

    def test_remove_element(self):
        nums1 = [3,2,2,3]
        k1 = remove_element(nums1, 3)
        self.assertEqual(k1, 2)
        self.assertEqual(nums1[:k1], [2,2])
        
        nums2 = [0,1,2,2,3,0,4,2]
        k2 = remove_element(nums2, 2)
        self.assertEqual(k2, 5)
        self.assertEqual(sorted(nums2[:k2]), [0,0,1,3,4])

    def test_two_sum_sorted(self):
        self.assertEqual(two_sum_sorted([2,7,11,15], 9), [1,2])
        self.assertEqual(two_sum_sorted([2,3,4], 6), [1,3])

    def test_find_max_consecutive_ones(self):
        self.assertEqual(find_max_consecutive_ones([1,1,0,1,1,1]), 3)
        self.assertEqual(find_max_consecutive_ones([1,0,1,1,0,1]), 2)

    def test_min_subarray_len(self):
        self.assertEqual(min_subarray_len(7, [2,3,1,2,4,3]), 2)
        self.assertEqual(min_subarray_len(4, [1,4,4]), 1)
        self.assertEqual(min_subarray_len(11, [1,1,1,1,1,1,1,1]), 0)

    def test_get_pascal_row(self):
        self.assertEqual(get_pascal_row(3), [1,3,3,1])
        self.assertEqual(get_pascal_row(0), [1])
        self.assertEqual(get_pascal_row(1), [1,1])
        self.assertEqual(get_pascal_row(4), [1,4,6,4,1])

    def test_reverse_words(self):
        self.assertEqual(reverse_words("the sky is blue"), "blue is sky the")
        self.assertEqual(reverse_words("  hello world  "), "world hello")
        self.assertEqual(reverse_words("a good   example"), "example good a")

    def test_reverse_words_iii(self):
        self.assertEqual(reverse_words_iii("Let's take LeetCode contest"), 
                         "s'teL ekat edoCteeL tsetnoc")
        self.assertEqual(reverse_words_iii("Mr Ding"), "rM gniD")

    def test_remove_duplicates(self):
        nums1 = [1,1,2]
        k1 = remove_duplicates(nums1)
        self.assertEqual(k1, 2)
        self.assertEqual(nums1[:k1], [1,2])
        
        nums2 = [0,0,1,1,1,2,2,3,3,4]
        k2 = remove_duplicates(nums2)
        self.assertEqual(k2, 5)
        self.assertEqual(nums2[:k2], [0,1,2,3,4])

    def test_move_zeroes(self):
        nums1 = [0,1,0,3,12]
        move_zeroes(nums1)
        self.assertEqual(nums1, [1,3,12,0,0])
        
        nums2 = [0]
        move_zeroes(nums2)
        self.assertEqual(nums2, [0])
        
        nums3 = [1,2,3,4,5]
        move_zeroes(nums3)
        self.assertEqual(nums3, [1,2,3,4,5])
        
        nums4 = [0,0,0,1,2,3]
        move_zeroes(nums4)
        self.assertEqual(nums4, [1,2,3,0,0,0])


# ======================================================
# main() Runner
# ======================================================

def main():
    print("Sample Runs:\n")

    print("1. Find Pivot Index:")
    print(f"   nums = [1,7,3,6,5,6] -> {pivot_index([1,7,3,6,5,6])}")
    
    print("\n2. Largest Number At Least Twice of Others:")
    print(f"   nums = [3,6,1,0] -> {dominant_index([3,6,1,0])}")
    
    print("\n3. Plus One:")
    print(f"   digits = [1,2,3] -> {plus_one([1,2,3])}")
    
    print("\n4. Diagonal Traverse:")
    print(f"   matrix = [[1,2,3],[4,5,6],[7,8,9]]")
    print(f"   result = {find_diagonal_order([[1,2,3],[4,5,6],[7,8,9]])}")
    
    print("\n5. Spiral Matrix:")
    print(f"   matrix = [[1,2,3],[4,5,6],[7,8,9]]")
    print(f"   result = {spiral_order([[1,2,3],[4,5,6],[7,8,9]])}")
    
    print("\n6. Pascal's Triangle (5 rows):")
    print(f"   {generate_pascals_triangle(5)}")
    
    print("\n7. Add Binary:")
    print(f"   '1010' + '1011' = {add_binary('1010', '1011')}")
    
    print("\n8. Longest Common Prefix:")
    print(f"   ['flower','flow','flight'] -> '{longest_common_prefix(['flower','flow','flight'])}'")
    
    print("\n9. strStr():")
    print(f"   'hello' with 'll' -> {str_str('hello', 'll')}")
    
    print("\n10. Reverse String:")
    s = ["h","e","l","l","o"]
    reverse_string(s)
    print(f"   'hello' -> {s}")
    
    print("\n11. Remove Element:")
    nums = [3,2,2,3]
    k = remove_element(nums, 3)
    print(f"   nums = [3,2,2,3], val = 3 -> k = {k}, nums[:{k}] = {nums[:k]}")
    
    print("\n12. Two Sum II - Input Sorted:")
    print(f"   [2,7,11,15], target=9 -> {two_sum_sorted([2,7,11,15], 9)}")
    
    print("\n13. Max Consecutive Ones:")
    print(f"   [1,1,0,1,1,1] -> {find_max_consecutive_ones([1,1,0,1,1,1])}")
    
    print("\n14. Minimum Size Subarray Sum:")
    print(f"   target=7, nums=[2,3,1,2,4,3] -> {min_subarray_len(7, [2,3,1,2,4,3])}")
    
    print("\n15. Pascal's Triangle II (row 3):")
    print(f"   rowIndex=3 -> {get_pascal_row(3)}")
    
    print("\n16. Reverse Words:")
    print(f"   'the sky is blue' -> '{reverse_words('the sky is blue')}'")
    
    print("\n17. Reverse Words III:")
    print(f"   \"Let's take LeetCode contest\" -> \"{reverse_words_iii("Let's take LeetCode contest")}\"")
    
    print("\n18. Remove Duplicates from Sorted Array:")
    nums = [0,0,1,1,1,2,2,3,3,4]
    k = remove_duplicates(nums)
    print(f"   [0,0,1,1,1,2,2,3,3,4] -> k = {k}, nums[:{k}] = {nums[:k]}")
    
    print("\n19. Move Zeroes:")
    nums = [0,1,0,3,12]
    move_zeroes(nums)
    print(f"   [0,1,0,3,12] -> {nums}")

    print("\nRunning Unit Tests...\n")
    unittest.main(exit=False, verbosity=2)


if __name__ == "__main__":
    main()