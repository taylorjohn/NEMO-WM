"""
Build LeetCode Pattern Knowledge Graph for Nemo WM
-------------------------------------------------
This script:
1. Loads the neenza/leetcode-problems dataset (local or downloaded)
2. Maps each problem to DSA patterns using community taxonomies + keyword heuristics
3. Extracts diagnostic triggers from problem descriptions
4. Aggregates into pattern cards (nodes) and relationships (edges)
5. Outputs a JSON knowledge graph suitable for Nemo WM integration
"""

import json
import os
import re
import urllib.request
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# =============================================================================
# Configuration
# =============================================================================

NEENZA_DATASET_URL = "https://raw.githubusercontent.com/neenza/leetcode-problems/master/merged_problems.json"
LOCAL_DATASET_PATH = "merged_problems.json"
OUTPUT_JSON = "dsa_pattern_kg.json"
OUTPUT_NEMO_FRIENDLY = "nemo_pattern_cards.json"

# Minimum confidence to assign a pattern via keyword matching
KEYWORD_CONFIDENCE_THRESHOLD = 2  # number of keyword hits required

# =============================================================================
# Step 1: Load LeetCode Dataset
# =============================================================================

def download_dataset(local_path: str) -> None:
    """Download the dataset, trying multiple known URLs for robustness."""
    urls_to_try = [
        "https://raw.githubusercontent.com/neenza/leetcode-problems/master/merged_problems.json",
        "https://raw.githubusercontent.com/neenza/leetcode-problems/main/leetcode_problems.json",
        "https://raw.githubusercontent.com/kaushik-bhat/Leetcode-problem-scraper/main/problems.json",
        "https://raw.githubusercontent.com/DoLeetCode/leetcode-json/main/leetcode.json"
    ]
    
    for try_url in urls_to_try:
        try:
            print(f"Attempting download from: {try_url}")
            req = urllib.request.Request(try_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Download successful from {try_url}")
            return
        except urllib.error.HTTPError as e:
            print(f"  Failed: HTTP Error {e.code} for {try_url}")
            continue
        except Exception as e:
            print(f"  Failed: {e} for {try_url}")
            continue
    
    raise RuntimeError("All download URLs failed. Please check your internet connection or the repository's status.")

def load_problems(filepath: str) -> List[Dict]:
    """Load problems from JSON file, handling various structures."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different possible structures
    if isinstance(data, list):
        print(f"Loaded {len(data)} problems from list structure")
        return data
    elif isinstance(data, dict):
        # Check common keys that contain the problem list
        possible_keys = ['problems', 'data', 'items', 'questions', 'leetcode_problems', 'merged_problems']
        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                print(f"Found {len(data[key])} problems under key: '{key}'")
                return data[key]
        
        # If no known key, look for the first list value containing dicts with problem-like fields
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and ('title' in value[0] or 'id' in value[0] or 'name' in value[0]):
                    print(f"Using key: '{key}' with {len(value)} items")
                    return value
        
        # If each value is a problem dict, return values
        if all(isinstance(v, dict) for v in data.values()):
            print(f"Converting dictionary with {len(data)} values to list")
            return list(data.values())
    
    # If we get here, print debug info
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Available keys: {list(data.keys())[:20]}")
        # Try to find any nested list
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        print(f"Found nested list at: {key}.{subkey}")
    raise ValueError(f"Unexpected JSON structure. Cannot find problem list.")

def normalize_problem(p: Dict) -> Dict:
    """Normalize problem fields to consistent names."""
    normalized = {}
    
    # ID
    normalized['id'] = p.get('id') or p.get('questionId') or p.get('problemId') or p.get('frontendQuestionId')
    
    # Title
    normalized['title'] = p.get('title') or p.get('questionTitle') or p.get('name') or p.get('problemName') or ''
    
    # Description
    normalized['description'] = p.get('description') or p.get('content') or p.get('question') or p.get('problemDescription') or ''
    
    # Difficulty
    difficulty = p.get('difficulty') or p.get('level') or ''
    if isinstance(difficulty, (int, float)):
        difficulty = {1: 'Easy', 2: 'Medium', 3: 'Hard'}.get(int(difficulty), 'Unknown')
    normalized['difficulty'] = difficulty
    
    # Topics/Tags
    topics = p.get('topics') or p.get('topicTags') or p.get('tags') or p.get('categories') or []
    if isinstance(topics, str):
        topics = [t.strip() for t in topics.split(',')]
    normalized['topics'] = topics
    
    # URL slug
    normalized['slug'] = p.get('slug') or p.get('titleSlug') or p.get('url') or ''
    
    return normalized

# =============================================================================
# Step 2: DSA Pattern Mapping (Community Taxonomies + Keyword Rules)
# =============================================================================

# Core DSA patterns with their diagnostic triggers (keywords/phrases)
PATTERN_KEYWORDS = {
    "Two Pointers": {
        "keywords": ["two pointer", "left and right", "start and end", "pair", "triplet",
                     "sorted array", "in-place", "remove duplicates", "move zeroes"],
        "constraints": ["sorted"]
    },
    "Sliding Window": {
        "keywords": ["subarray", "substring", "contiguous", "window", "consecutive",
                     "k elements", "maximum sum subarray", "minimum window"],
        "constraints": ["k", "window size"]
    },
    "Binary Search": {
        "keywords": ["binary search", "sorted array", "log n", "search insert",
                     "find minimum", "peak element", "rotated sorted"],
        "constraints": ["O(log n)"]
    },
    "BFS": {
        "keywords": ["level order", "shortest path", "queue", "breadth first",
                     "minimum steps", "neighbors", "grid traversal"],
        "constraints": ["unweighted", "shortest"]
    },
    "DFS": {
        "keywords": ["depth first", "backtracking", "recursive", "explore all",
                     "permutations", "combinations", "subsets", "path sum"],
        "constraints": ["all possible"]
    },
    "Dynamic Programming": {
        "keywords": ["optimal", "maximum", "minimum", "count ways", "number of ways",
                     "climbing stairs", "knapsack", "longest", "subsequence"],
        "constraints": ["overlapping subproblems"]
    },
    "Graph": {
        "keywords": ["graph", "nodes", "edges", "adjacency", "topological sort",
                     "union find", "disjoint set", "connected components"],
        "constraints": ["graph"]
    },
    "Tree": {
        "keywords": ["tree", "binary tree", "bst", "inorder", "preorder", "postorder",
                     "lowest common ancestor", "depth", "height"],
        "constraints": ["tree"]
    },
    "Heap / Priority Queue": {
        "keywords": ["kth", "top k", "largest", "smallest", "priority queue",
                     "merge k", "median"],
        "constraints": ["k"]
    },
    "Backtracking": {
        "keywords": ["backtrack", "all possible", "generate all", "n-queens",
                     "sudoku", "combinations", "permutations"],
        "constraints": ["generate"]
    },
    "Greedy": {
        "keywords": ["greedy", "maximum profit", "minimum coins", "interval scheduling",
                     "jump game", "gas station"],
        "constraints": ["local optimal"]
    },
    "Hash Table": {
        "keywords": ["hash", "map", "dictionary", "two sum", "first unique",
                     "duplicate", "frequency", "count"],
        "constraints": ["O(1) lookup"]
    },
    "Stack / Queue": {
        "keywords": ["stack", "queue", "monotonic", "valid parentheses", "next greater",
                     "daily temperatures", "evaluate expression"],
        "constraints": ["LIFO", "FIFO"]
    },
    "Linked List": {
        "keywords": ["linked list", "node", "next", "reverse", "cycle", "merge"],
        "constraints": ["linked list"]
    },
    "Bit Manipulation": {
        "keywords": ["bit", "xor", "and", "or", "bitwise", "single number", "power of two"],
        "constraints": ["bit manipulation"]
    },
    "Math": {
        "keywords": ["prime", "factorial", "gcd", "lcm", "permutation", "combination",
                     "divisor", "multiple"],
        "constraints": ["math"]
    },
    "String": {
        "keywords": ["string", "palindrome", "anagram", "substring", "pattern matching"],
        "constraints": ["string"]
    },
    "Array": {
        "keywords": ["array", "matrix", "rotate", "transpose", "spiral", "diagonal"],
        "constraints": ["array"]
    },
    "Union Find": {
        "keywords": ["union find", "disjoint set", "connected components", "find", "union"],
        "constraints": ["connectivity", "cycle detection"]
    },
    "Trie": {
        "keywords": ["trie", "prefix tree", "word search", "autocomplete"],
        "constraints": ["prefix"]
    },
    "Segment Tree": {
        "keywords": ["segment tree", "range query", "range sum", "range minimum"],
        "constraints": ["range update"]
    },
    "Binary Indexed Tree": {
        "keywords": ["fenwick", "binary indexed tree", "prefix sum", "range sum"],
        "constraints": ["point update"]
    }
}

# Manual mapping from problem title to pattern (curated from public taxonomies)
TITLE_TO_PATTERN = {
    "Two Sum": "Hash Table",
    "Add Two Numbers": "Linked List",
    "Longest Substring Without Repeating Characters": "Sliding Window",
    "Median of Two Sorted Arrays": "Binary Search",
    "Longest Palindromic Substring": "Dynamic Programming",
    "Container With Most Water": "Two Pointers",
    "3Sum": "Two Pointers",
    "Remove Nth Node From End of List": "Two Pointers",
    "Valid Parentheses": "Stack / Queue",
    "Merge Two Sorted Lists": "Linked List",
    "Generate Parentheses": "Backtracking",
    "Merge k Sorted Lists": "Heap / Priority Queue",
    "Swap Nodes in Pairs": "Linked List",
    "Reverse Nodes in k-Group": "Linked List",
    "Remove Duplicates from Sorted Array": "Two Pointers",
    "Remove Element": "Two Pointers",
    "Find the Index of the First Occurrence in a String": "String",
    "Divide Two Integers": "Math",
    "Substring with Concatenation of All Words": "Sliding Window",
    "Next Permutation": "Array",
    "Longest Valid Parentheses": "Dynamic Programming",
    "Search in Rotated Sorted Array": "Binary Search",
    "Find First and Last Position of Element in Sorted Array": "Binary Search",
    "Search Insert Position": "Binary Search",
    "Valid Sudoku": "Hash Table",
    "Sudoku Solver": "Backtracking",
    "Count and Say": "String",
    "Combination Sum": "Backtracking",
    "Combination Sum II": "Backtracking",
    "First Missing Positive": "Array",
    "Trapping Rain Water": "Two Pointers",
    "Multiply Strings": "Math",
    "Wildcard Matching": "Dynamic Programming",
    "Jump Game II": "Greedy",
    "Permutations": "Backtracking",
    "Permutations II": "Backtracking",
    "Rotate Image": "Array",
    "Group Anagrams": "Hash Table",
    "Pow(x, n)": "Math",
    "N-Queens": "Backtracking",
    "N-Queens II": "Backtracking",
    "Maximum Subarray": "Dynamic Programming",
    "Spiral Matrix": "Array",
    "Jump Game": "Greedy",
    "Merge Intervals": "Array",
    "Insert Interval": "Array",
    "Length of Last Word": "String",
    "Spiral Matrix II": "Array",
    "Permutation Sequence": "Math",
    "Rotate List": "Linked List",
    "Unique Paths": "Dynamic Programming",
    "Unique Paths II": "Dynamic Programming",
    "Minimum Path Sum": "Dynamic Programming",
    "Valid Number": "String",
    "Plus One": "Array",
    "Add Binary": "Math",
    "Text Justification": "String",
    "Sqrt(x)": "Binary Search",
    "Climbing Stairs": "Dynamic Programming",
    "Simplify Path": "Stack / Queue",
    "Edit Distance": "Dynamic Programming",
    "Set Matrix Zeroes": "Array",
    "Search a 2D Matrix": "Binary Search",
    "Sort Colors": "Two Pointers",
    "Minimum Window Substring": "Sliding Window",
    "Subsets": "Backtracking",
    "Word Search": "Backtracking",
    "Largest Rectangle in Histogram": "Stack / Queue",
    "Maximal Rectangle": "Dynamic Programming",
    "Binary Tree Inorder Traversal": "Tree",
    "Unique Binary Search Trees II": "Dynamic Programming",
    "Unique Binary Search Trees": "Dynamic Programming",
    "Interleaving String": "Dynamic Programming",
    "Validate Binary Search Tree": "Tree",
    "Recover Binary Search Tree": "Tree",
    "Same Tree": "Tree",
    "Symmetric Tree": "Tree",
    "Binary Tree Level Order Traversal": "BFS",
    "Binary Tree Zigzag Level Order Traversal": "BFS",
    "Maximum Depth of Binary Tree": "Tree",
    "Construct Binary Tree from Preorder and Inorder Traversal": "Tree",
    "Construct Binary Tree from Inorder and Postorder Traversal": "Tree",
    "Binary Tree Level Order Traversal II": "BFS",
    "Convert Sorted Array to Binary Search Tree": "Tree",
    "Convert Sorted List to Binary Search Tree": "Tree",
    "Balanced Binary Tree": "Tree",
    "Minimum Depth of Binary Tree": "Tree",
    "Path Sum": "Tree",
    "Path Sum II": "Backtracking",
    "Flatten Binary Tree to Linked List": "Tree",
    "Distinct Subsequences": "Dynamic Programming",
    "Populating Next Right Pointers in Each Node": "BFS",
    "Populating Next Right Pointers in Each Node II": "BFS",
    "Pascal's Triangle": "Dynamic Programming",
    "Pascal's Triangle II": "Dynamic Programming",
    "Triangle": "Dynamic Programming",
    "Best Time to Buy and Sell Stock": "Array",
    "Best Time to Buy and Sell Stock II": "Greedy",
    "Best Time to Buy and Sell Stock III": "Dynamic Programming",
    "Binary Tree Maximum Path Sum": "Tree",
    "Valid Palindrome": "Two Pointers",
    "Word Ladder": "BFS",
    "Word Ladder II": "BFS",
    "Longest Consecutive Sequence": "Hash Table",
    "Sum Root to Leaf Numbers": "Tree",
    "Surrounded Regions": "DFS",
    "Palindrome Partitioning": "Backtracking",
    "Palindrome Partitioning II": "Dynamic Programming",
    "Clone Graph": "Graph",
    "Gas Station": "Greedy",
    "Candy": "Greedy",
    "Single Number": "Bit Manipulation",
    "Single Number II": "Bit Manipulation",
    "Copy List with Random Pointer": "Linked List",
    "Word Break": "Dynamic Programming",
    "Word Break II": "Dynamic Programming",
    "Linked List Cycle": "Two Pointers",
    "Linked List Cycle II": "Two Pointers",
    "Reorder List": "Linked List",
    "Binary Tree Preorder Traversal": "Tree",
    "Binary Tree Postorder Traversal": "Tree",
    "LRU Cache": "Hash Table",
    "Insertion Sort List": "Linked List",
    "Sort List": "Linked List",
    "Max Points on a Line": "Hash Table",
    "Evaluate Reverse Polish Notation": "Stack / Queue",
    "Reverse Words in a String": "String",
    "Maximum Product Subarray": "Dynamic Programming",
    "Find Minimum in Rotated Sorted Array": "Binary Search",
    "Find Minimum in Rotated Sorted Array II": "Binary Search",
    "Min Stack": "Stack / Queue",
    "Intersection of Two Linked Lists": "Linked List",
    "Find Peak Element": "Binary Search",
    "Compare Version Numbers": "String",
    "Fraction to Recurring Decimal": "Hash Table",
    "Two Sum II - Input Array Is Sorted": "Two Pointers",
    "Excel Sheet Column Title": "Math",
    "Majority Element": "Array",
    "Excel Sheet Column Number": "Math",
    "Factorial Trailing Zeroes": "Math",
    "Binary Search Tree Iterator": "Tree",
    "Dungeon Game": "Dynamic Programming",
    "Largest Number": "String",
    "Repeated DNA Sequences": "Hash Table",
    "Best Time to Buy and Sell Stock IV": "Dynamic Programming",
    "Rotate Array": "Array",
    "Reverse Bits": "Bit Manipulation",
    "Number of 1 Bits": "Bit Manipulation",
    "House Robber": "Dynamic Programming",
    "Binary Tree Right Side View": "BFS",
    "Number of Islands": "DFS",
    "Bitwise AND of Numbers Range": "Bit Manipulation",
    "Happy Number": "Two Pointers",
    "Remove Linked List Elements": "Linked List",
    "Count Primes": "Math",
    "Isomorphic Strings": "Hash Table",
    "Reverse Linked List": "Linked List",
    "Course Schedule": "Graph",
    "Course Schedule II": "Graph",
    "Implement Trie (Prefix Tree)": "Trie",
    "Minimum Size Subarray Sum": "Sliding Window",
    "Word Search II": "Backtracking",
    "House Robber II": "Dynamic Programming",
    "Shortest Palindrome": "String",
    "Kth Largest Element in an Array": "Heap / Priority Queue",
    "Combination Sum III": "Backtracking",
    "Contains Duplicate": "Hash Table",
    "Contains Duplicate II": "Hash Table",
    "Contains Duplicate III": "Sliding Window",
    "Maximal Square": "Dynamic Programming",
    "Count Complete Tree Nodes": "Tree",
    "Rectangle Area": "Math",
    "Basic Calculator": "Stack / Queue",
    "Implement Stack using Queues": "Stack / Queue",
    "Implement Queue using Stacks": "Stack / Queue",
    "Palindrome Linked List": "Two Pointers",
    "Lowest Common Ancestor of a Binary Search Tree": "Tree",
    "Lowest Common Ancestor of a Binary Tree": "Tree",
    "Delete Node in a Linked List": "Linked List",
    "Product of Array Except Self": "Array",
    "Sliding Window Maximum": "Sliding Window",
    "Search a 2D Matrix II": "Binary Search",
    "Valid Anagram": "Hash Table",
    "Binary Tree Paths": "Tree",
    "Add Digits": "Math",
    "Single Number III": "Bit Manipulation",
    "Missing Number": "Bit Manipulation",
    "H-Index II": "Binary Search",
    "First Bad Version": "Binary Search",
    "Perfect Squares": "Dynamic Programming",
    "Move Zeroes": "Two Pointers",
    "Find the Duplicate Number": "Two Pointers",
    "Game of Life": "Array",
    "Word Pattern": "Hash Table",
    "Nim Game": "Math",
    "Find Median from Data Stream": "Heap / Priority Queue",
    "Serialize and Deserialize Binary Tree": "Tree",
    "Longest Increasing Subsequence": "Dynamic Programming",
    "Remove Invalid Parentheses": "BFS",
    "Range Sum Query - Immutable": "Dynamic Programming",
    "Range Sum Query 2D - Immutable": "Dynamic Programming",
    "Number of Islands II": "Union Find",
    "Additive Number": "Backtracking",
    "Range Sum Query - Mutable": "Segment Tree",
    "Best Time to Buy and Sell Stock with Cooldown": "Dynamic Programming",
    "Minimum Height Trees": "Graph",
    "Burst Balloons": "Dynamic Programming",
    "Super Ugly Number": "Dynamic Programming",
    "Count of Smaller Numbers After Self": "Binary Search",
    "Remove Duplicate Letters": "Stack / Queue",
    "Maximum Product of Word Lengths": "Bit Manipulation",
    "Bulb Switcher": "Math",
    "Coin Change": "Dynamic Programming",
    "Power of Three": "Math",
    "Odd Even Linked List": "Linked List",
    "Longest Increasing Path in a Matrix": "DFS",
    "Patching Array": "Greedy",
    "Verify Preorder Serialization of a Binary Tree": "Tree",
    "Reconstruct Itinerary": "DFS",
    "Largest Divisible Subset": "Dynamic Programming",
    "Increasing Triplet Subsequence": "Greedy",
    "Palindrome Pairs": "Hash Table",
    "House Robber III": "Dynamic Programming",
    "Counting Bits": "Dynamic Programming",
    "Flatten Nested List Iterator": "Stack / Queue",
    "Power of Four": "Bit Manipulation",
    "Integer Break": "Dynamic Programming",
    "Reverse String": "Two Pointers",
    "Reverse Vowels of a String": "Two Pointers",
    "Top K Frequent Elements": "Heap / Priority Queue",
    "Intersection of Two Arrays": "Hash Table",
    "Intersection of Two Arrays II": "Hash Table",
    "Russian Doll Envelopes": "Dynamic Programming",
    "Sort Characters By Frequency": "Heap / Priority Queue",
    "4Sum II": "Hash Table",
    "Assign Cookies": "Greedy",
    "132 Pattern": "Stack / Queue",
    "Circular Array Loop": "Two Pointers",
    "Poor Pigs": "Math",
    "Repeated Substring Pattern": "String",
    "LFU Cache": "Hash Table",
    "Hamming Distance": "Bit Manipulation",
    "Island Perimeter": "Array",
    "Can I Win": "Dynamic Programming",
    "Validate IP Address": "String",
    "Encode and Decode TinyURL": "Hash Table",
    "Concatenated Words": "Dynamic Programming",
    "Matchsticks to Square": "Backtracking",
    "Ones and Zeroes": "Dynamic Programming",
    "Heaters": "Binary Search",
    "Number Complement": "Bit Manipulation",
    "Total Hamming Distance": "Bit Manipulation",
    "Sliding Window Median": "Sliding Window",
    "License Key Formatting": "String",
    "Max Consecutive Ones": "Array",
    "Predict the Winner": "Dynamic Programming",
    "Increasing Subsequences": "Backtracking",
    "Target Sum": "Dynamic Programming",
    "Teemo Attacking": "Array",
    "Next Greater Element I": "Stack / Queue",
    "Next Greater Element II": "Stack / Queue",
    "Find Mode in Binary Search Tree": "Tree",
    "IPO": "Heap / Priority Queue",
    "Most Frequent Subtree Sum": "Tree",
    "Fibonacci Number": "Math",
    "Find Bottom Left Tree Value": "Tree",
    "Freedom Trail": "Dynamic Programming",
    "Find Largest Value in Each Tree Row": "Tree",
    "Longest Palindromic Subsequence": "Dynamic Programming",
    "Super Washing Machines": "Greedy",
    "Coin Change 2": "Dynamic Programming",
    "Detect Capital": "String",
    "Longest Uncommon Subsequence I": "String",
    "Continuous Subarray Sum": "Hash Table",
    "Longest Word in Dictionary through Deleting": "Two Pointers",
    "Contiguous Array": "Hash Table",
    "Beautiful Arrangement": "Backtracking",
    "Minesweeper": "DFS",
    "Minimum Absolute Difference in BST": "Tree",
    "K-diff Pairs in an Array": "Two Pointers",
    "Encode and Decode Strings": "String",
    "Convert BST to Greater Tree": "Tree",
    "Reverse String II": "Two Pointers",
    "01 Matrix": "BFS",
    "Diameter of Binary Tree": "Tree",
    "Remove Boxes": "Dynamic Programming",
    "Friend Circles": "Union Find",
    "Binary Tree Longest Consecutive Sequence": "Tree",
    "Maximum Product of Three Numbers": "Array",
    "Add One Row to Tree": "Tree",
    "Maximum Swap": "Math",
    "Second Minimum Node In a Binary Tree": "Tree",
    "Count Univalue Subtrees": "Tree",
    "Longest Univalue Path": "Tree",
    "Stickers to Spell Word": "Dynamic Programming",
    "Number of Distinct Islands": "DFS",
    "Minimum ASCII Delete Sum for Two Strings": "Dynamic Programming",
    "Subarray Product Less Than K": "Sliding Window",
    "Best Time to Buy and Sell Stock with Transaction Fee": "Dynamic Programming",
    "Maximum Length of Repeated Subarray": "Dynamic Programming",
    "Longest Word in Dictionary": "Trie",
    "Kth Smallest Element in a Sorted Matrix": "Binary Search",
    "Find the Derangement of An Array": "Dynamic Programming",
    "Maximum Average Subarray I": "Sliding Window",
    "Set Mismatch": "Hash Table",
    "Maximum Length of Pair Chain": "Dynamic Programming",
    "Palindromic Substrings": "Dynamic Programming",
    "Replace Words": "Trie",
    "Dota2 Senate": "Greedy",
    "2 Keys Keyboard": "Dynamic Programming",
    "Find Duplicate File in System": "Hash Table",
    "Construct String from Binary Tree": "Tree",
    "Find Duplicate Subtrees": "Tree",
    "Two Sum IV - Input is a BST": "Tree",
    "Maximum Binary Tree": "Tree",
    "Print Binary Tree": "Tree",
    "Robot Return to Origin": "String",
    "Find K Closest Elements": "Binary Search",
    "Split Array into Consecutive Subsequences": "Greedy",
    "Image Smoother": "Array",
    "Maximum Width of Binary Tree": "Tree",
    "Strange Printer": "Dynamic Programming",
    "Non-decreasing Array": "Array",
    "Trim a Binary Search Tree": "Tree",
    "Number of Longest Increasing Subsequence": "Dynamic Programming",
    "Longest Continuous Increasing Subsequence": "Array",
    "Cut Off Trees for Golf Event": "BFS",
    "Implement Magic Dictionary": "Trie",
    "Map Sum Pairs": "Trie",
    "Valid Parenthesis String": "Dynamic Programming",
    "24 Game": "Backtracking",
    "Valid Palindrome II": "Two Pointers",
    "Baseball Game": "Stack / Queue",
    "Redundant Connection": "Union Find",
    "Redundant Connection II": "Union Find",
    "Repeated String Match": "String",
    "Maximum Sum of 3 Non-Overlapping Subarrays": "Dynamic Programming",
    "Employee Importance": "BFS",
    "Top K Frequent Words": "Heap / Priority Queue",
    "Maximum Binary Tree II": "Tree",
    "Find Anagram Mappings": "Hash Table",
    "Insert into a Binary Search Tree": "Tree",
    "Search in a Binary Search Tree": "Tree",
    "Kth Largest Element in a Stream": "Heap / Priority Queue",
    "Binary Search": "Binary Search",
    "Min Cost Climbing Stairs": "Dynamic Programming",
    "Find Smallest Letter Greater Than Target": "Binary Search",
    "Network Delay Time": "Graph",
    "Open the Lock": "BFS",
    "Cracking the Safe": "DFS",
    "Reach a Number": "Math",
    "Champagne Tower": "Dynamic Programming",
    "Minimum Swaps To Make Sequences Increasing": "Dynamic Programming",
    "Find Eventual Safe States": "Graph",
    "Bricks Falling When Hit": "Union Find",
    "Largest Plus Sign": "Dynamic Programming",
    "Reorganize String": "Heap / Priority Queue",
    "Max Chunks To Make Sorted": "Array",
    "Jewels and Stones": "Hash Table",
    "Sliding Puzzle": "BFS",
    "Minimize Max Distance to Gas Station": "Binary Search",
    "Global and Local Inversions": "Array",
    "Swap Adjacent in LR String": "Two Pointers",
    "Swim in Rising Water": "Binary Search",
    "K-th Symbol in Grammar": "Math",
    "Reaching Points": "Math",
    "Rabbits in Forest": "Math",
    "Minimum Distance Between BST Nodes": "Tree",
    "Letter Case Permutation": "Backtracking",
    "Is Graph Bipartite?": "Graph",
    "K-th Smallest Prime Fraction": "Binary Search",
    "Cheapest Flights Within K Stops": "Graph",
    "Rotated Digits": "Math",
    "Escape The Ghosts": "Math",
    "Domino and Tromino Tiling": "Dynamic Programming",
    "Custom Sort String": "String",
    "Number of Matching Subsequences": "String",
    "Preimage Size of Factorial Zeroes Function": "Binary Search",
    "Valid Tic-Tac-Toe State": "Array",
    "Number of Subarrays with Bounded Maximum": "Two Pointers",
    "Rotate String": "String",
    "All Paths From Source to Target": "DFS",
    "Smallest Rotation with Highest Score": "Array",
    "Binary Tree Pruning": "Tree",
    "Bus Routes": "BFS",
    "Ambiguous Coordinates": "String",
    "Linked List Components": "Linked List",
    "Race Car": "BFS",
    "Most Common Word": "String",
    "Shortest Distance to a Character": "Two Pointers",
    "Card Flipping Game": "Hash Table",
    "Binary Trees With Factors": "Dynamic Programming"
}

def extract_patterns_from_description(description: str) -> List[Tuple[str, int]]:
    """Use keyword matching to guess patterns from problem description."""
    if not description:
        return []
    description_lower = description.lower()
    pattern_scores = defaultdict(int)
    for pattern, data in PATTERN_KEYWORDS.items():
        for kw in data["keywords"]:
            if kw.lower() in description_lower:
                pattern_scores[pattern] += 1
        for constraint in data.get("constraints", []):
            if constraint.lower() in description_lower:
                pattern_scores[pattern] += 1
    return [(p, s) for p, s in pattern_scores.items() if s >= KEYWORD_CONFIDENCE_THRESHOLD]

def assign_patterns(problem: Dict) -> List[str]:
    """Assign one or more DSA patterns to a problem."""
    patterns = set()
    
    # Normalize problem fields
    p = normalize_problem(problem)
    title = p.get('title', '')
    description = p.get('description', '')
    topics = p.get('topics', [])
    
    # Check manual mapping first
    if title in TITLE_TO_PATTERN:
        patterns.add(TITLE_TO_PATTERN[title])
    
    # Check for pattern in title
    for pattern in PATTERN_KEYWORDS:
        if pattern.lower() in title.lower():
            patterns.add(pattern)
    
    # Keyword extraction from description
    if description:
        for pat, score in extract_patterns_from_description(description):
            patterns.add(pat)
    
    # Check topics field
    for topic in topics:
        topic_lower = topic.lower()
        for pattern in PATTERN_KEYWORDS:
            if pattern.lower() in topic_lower:
                patterns.add(pattern)
    
    # If no pattern assigned, try to infer from difficulty and common patterns
    if not patterns:
        # Default to Array/Hash Table for easy problems, DP/Graph for hard
        difficulty = p.get('difficulty', '')
        if difficulty == 'Easy':
            patterns.add('Array')
        elif difficulty == 'Hard':
            patterns.add('Dynamic Programming')
        else:
            patterns.add('Array')
    
    return list(patterns)

# =============================================================================
# Step 3: Extract Diagnostic Triggers per Pattern
# =============================================================================
def extract_diagnostic_triggers(pattern: str, problems: List[Dict]) -> List[str]:
    """
    Aggregate common keywords from problems assigned to this pattern
    to serve as diagnostic triggers.
    """
    all_descs = []
    for prob in problems:
        if pattern in prob.get("assigned_patterns", []):
            p = normalize_problem(prob)
            desc = p.get("description", "")
            if desc:
                all_descs.append(desc.lower())
    
    if not all_descs:
        return PATTERN_KEYWORDS.get(pattern, {}).get("keywords", [])[:10]
    
    # Count word frequencies using Counter instead of defaultdict
    from collections import Counter
    word_counts = Counter()
    stopwords = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "is", "are", "given", "you", "return", "that", "this", "be", "if"}
    
    for desc in all_descs:
        words = re.findall(r'\b[a-z]{3,}\b', desc)
        for w in words:
            if w not in stopwords:
                word_counts[w] += 1
    
    # Filter to relevant words
    pattern_kw_set = set(PATTERN_KEYWORDS.get(pattern, {}).get("keywords", []))
    relevant_words = []
    for word, count in word_counts.most_common(50):
        if word in pattern_kw_set or any(kw in word for kw in pattern_kw_set):
            relevant_words.append(word)
    
    return relevant_words[:15]

# =============================================================================
# Step 4: Build Knowledge Graph (Pattern Cards + Relationships)
# =============================================================================

def build_pattern_cards(problems: List[Dict]) -> Dict[str, Dict]:
    """Aggregate problems into pattern cards."""
    pattern_to_problems = defaultdict(list)
    for p in problems:
        for pat in p.get("assigned_patterns", []):
            pattern_to_problems[pat].append(p)
    
    cards = {}
    for pattern, probs in pattern_to_problems.items():
        # Diagnostic triggers
        triggers = extract_diagnostic_triggers(pattern, problems)
        
        # Sample problem titles
        sample_titles = [normalize_problem(p).get('title', '') for p in probs[:15]]
        sample_titles = [t for t in sample_titles if t]
        
        pattern_id = f"dsa.{pattern.lower().replace(' ', '_').replace('/', '_')}"
        
        cards[pattern] = {
            "pattern_id": pattern_id,
            "name": pattern,
            "family": "dsa",
            "level": "primitive" if pattern in ["Two Pointers", "Sliding Window", "Binary Search", "Hash Table", "Stack / Queue", "Linked List", "Tree", "Heap / Priority Queue", "BFS", "DFS"] else "midlevel",
            "aliases": [],
            "core_invariant": f"Algorithmic pattern for {pattern}",
            "preconditions": [f"Problem exhibits characteristics of {pattern}"],
            "reusable_helpers": [f"implement_{pattern.lower().replace(' ', '_').replace('/', '_')}"],
            "arc_adaptations": [f"Apply {pattern} logic to grid/object transformations"],
            "diagnostic_triggers": triggers[:10],
            "parameter_slots": {
                "implementation": ["iterative", "recursive", "optimized"]
            },
            "synthetic_task_templates": [
                f"Create ARC task requiring {pattern} reasoning"
            ],
            "coverage_status": {
                "has_helper": False,
                "has_solver_family": False,
                "has_synthetic_generator": False,
                "has_diagnostic_trigger": True
            },
            "priority": 8,
            "related_patterns": [],
            "problem_count": len(probs),
            "sample_problems": sample_titles[:10],
            "keyword_triggers": PATTERN_KEYWORDS.get(pattern, {}).get("keywords", [])[:10]
        }
    return cards

def build_relationships(cards: Dict) -> List[Dict]:
    """Infer relationships between patterns based on keyword overlap."""
    relationships = []
    pattern_names = list(cards.keys())
    for i, p1 in enumerate(pattern_names):
        for p2 in pattern_names[i+1:]:
            common_kw = set(PATTERN_KEYWORDS.get(p1, {}).get("keywords", [])) & \
                       set(PATTERN_KEYWORDS.get(p2, {}).get("keywords", []))
            if common_kw:
                relationships.append({
                    "source": p1,
                    "target": p2,
                    "relation": "related_to",
                    "weight": len(common_kw)
                })
    return relationships

# =============================================================================
# Step 5: Output for Nemo WM
# =============================================================================

def output_nemo_friendly(cards: Dict, output_path: str):
    """Convert pattern cards to Nemo WM consumable format."""
    nemo_cards = []
    for pattern, card in cards.items():
        nemo_card = {
            "id": card["pattern_id"],
            "name": card["name"],
            "family": "dsa",
            "core_invariant": card["core_invariant"],
            "diagnostic_triggers": card["diagnostic_triggers"],
            "parameter_slots": card["parameter_slots"],
            "synthetic_task_templates": card["synthetic_task_templates"],
            "priority": card["priority"],
            "metadata": {
                "problem_count": card["problem_count"],
                "sample_problems": card["sample_problems"]
            }
        }
        nemo_cards.append(nemo_card)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nemo_cards, f, indent=2)
    print(f"Nemo-friendly pattern cards saved to {output_path}")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    # Step 1: Load dataset
    if not os.path.exists(LOCAL_DATASET_PATH):
        download_dataset(LOCAL_DATASET_PATH)
    else:
        print(f"Using existing dataset at {LOCAL_DATASET_PATH}")
    
    problems = load_problems(LOCAL_DATASET_PATH)
    print(f"Loaded {len(problems)} problems.")
    
    # Step 2: Assign patterns
    print("Assigning DSA patterns...")
    assigned_count = 0
    for p in problems:
        p["assigned_patterns"] = assign_patterns(p)
        if p["assigned_patterns"]:
            assigned_count += 1
    print(f"Assigned patterns to {assigned_count}/{len(problems)} problems.")
    
    # Step 3 & 4: Build pattern cards and relationships
    print("Building pattern cards...")
    cards = build_pattern_cards(problems)
    relationships = build_relationships(cards)
    print(f"Created {len(cards)} pattern cards.")
    
    # Output full knowledge graph
    kg_output = {
        "nodes": list(cards.values()),
        "edges": relationships,
        "metadata": {
            "total_problems": len(problems),
            "patterns_found": len(cards),
            "generated_at": __import__('datetime').datetime.now().isoformat()
        }
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(kg_output, f, indent=2)
    print(f"Full knowledge graph saved to {OUTPUT_JSON}")
    
    # Output Nemo-friendly version
    output_nemo_friendly(cards, OUTPUT_NEMO_FRIENDLY)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PATTERN SUMMARY")
    print('='*60)
    for pattern, card in sorted(cards.items(), key=lambda x: x[1]["problem_count"], reverse=True):
        triggers = ', '.join(card['diagnostic_triggers'][:5])
        print(f"\n{pattern} ({card['problem_count']} problems)")
        print(f"  ID: {card['pattern_id']}")
        print(f"  Triggers: {triggers}")
        print(f"  Sample: {card['sample_problems'][0] if card['sample_problems'] else 'N/A'}")

if __name__ == "__main__":
    main()