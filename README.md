# ArcadiaEngine 🎮

## 📋 Project Overview

ArcadiaEngine is a modular game engine framework that implements four core systems using various data structures and algorithmic paradigms:

- **Data Structures**: Hash tables, Skip lists, and Red-Black trees
- **Dynamic Programming**: Optimization problems for inventory management
- **Graph Algorithms**: Pathfinding and minimum spanning trees
- **Greedy Algorithms**: Task scheduling with constraints

## 🏗️ Architecture

### Part A: Data Structures

#### 1. PlayerTable (Double Hashing)
- **Implementation**: Hash table with double hashing collision resolution
- **Hash Functions**: 
  - `h1(key) = key % TABLE_SIZE`
  - `h2(key) = 1 + (key % (TABLE_SIZE - 1))`
- **Operations**: 
  - `insert(playerID, name)` - O(1) average case
  - `search(playerID)` - O(1) average case
- **Features**: Lazy deletion support

#### 2. Leaderboard (Skip List)
- **Implementation**: Probabilistic data structure with multiple levels
- **Ordering**: Descending by score, ascending by ID for ties
- **Operations**:
  - `addScore(playerID, score)` - O(log n) average
  - `removePlayer(playerID)` - O(log n) average
  - `getTopN(n)` - O(n) retrieval
- **Max Levels**: 16

#### 3. AuctionTree (Red-Black Tree)
- **Implementation**: Self-balancing binary search tree
- **Ordering**: Ascending by price, then by itemID for ties
- **Operations**:
  - `insertItem(itemID, price)` - O(log n)
  - `deleteItem(itemID)` - O(log n)
- **Features**: Maintains balance through rotations and color properties

### Part B: Inventory System (Dynamic Programming)

#### 1. Loot Splitting (Partition Problem)
- **Problem**: Minimize difference between two coin stacks
- **Algorithm**: Subset sum DP
- **Complexity**: O(n × sum/2)
- **Example**: `{1, 2, 4}` → minimum difference of 1

#### 2. Inventory Packer (0/1 Knapsack)
- **Problem**: Maximize value within weight capacity
- **Algorithm**: Space-optimized DP
- **Complexity**: O(n × capacity)
- **Example**: Capacity 10, items `{(1,10), (2,20), (3,30)}` → value 60

#### 3. Chat Decoder (String DP)
- **Problem**: Count possible decodings of a string
- **Rules**: "uu" or "nn" can be decoded as themselves or as "w"/"m"
- **Algorithm**: Linear DP with modulo arithmetic
- **Complexity**: O(n)
- **Example**: "uu" → 2 possibilities

### Part C: World Navigator (Graph Algorithms)

#### 1. Safe Passage (Path Existence)
- **Problem**: Check if path exists between two nodes
- **Algorithm**: Breadth-First Search (BFS)
- **Complexity**: O(V + E)
- **Graph Type**: Undirected

#### 2. The Bribe (Minimum Spanning Tree)
- **Problem**: Minimize total bribe cost to connect all cities
- **Algorithm**: Kruskal's algorithm with Union-Find
- **Complexity**: O(E log E)
- **Cost Formula**: `goldCost × goldRate + silverCost × silverRate`
- **Returns**: -1 if graph cannot be fully connected

#### 3. Teleporter Network (All-Pairs Shortest Path)
- **Problem**: Sum all shortest distances and return in binary
- **Algorithm**: Floyd-Warshall
- **Complexity**: O(V³)
- **Example**: Line graph with edges 0-1 (1), 1-2 (2) → sum = 6 → "110"

### Part D: Server Kernel (Greedy Algorithms)

#### Task Scheduler
- **Problem**: Schedule tasks with cooling time constraints
- **Constraint**: Same task must wait `n` intervals before running again
- **Algorithm**: Greedy frequency-based scheduling
- **Complexity**: O(m) where m is number of tasks
- **Example**: `{A, A, B}` with n=2 → 4 intervals (A → B → idle → A)

## 🚀 Getting Started

### Prerequisites

- C++ compiler with C++11 support or higher (g++, clang++)
- Make (optional)

### Compilation

```bash
# Compile all files
g++ -std=c++11 -o arcadia_test ArcadiaEngine.cpp main_test_students.cpp

# Run tests
./arcadia_test
```

### Project Structure

```
.
├── ArcadiaEngine.h          # Interface definitions
├── ArcadiaEngine.cpp        # Implementation
├── main_test_students.cpp   # Test suite
└── README.md               # This file
```

## 🧪 Testing

The project includes a comprehensive test suite covering:

- Basic functionality tests for all data structures
- Edge cases (empty inputs, single elements, ties)
- Algorithm correctness verification
- Example cases from assignment specifications

### Running Tests

```bash
./arcadia_test
```

Expected output:
```
TEST: PlayerTable: Insert 'Alice' and Search              [ PASS ]
TEST: Leaderboard: Add Scores & Get Top 1                 [ PASS ]
TEST: Leaderboard: Tie-Break (ID 10 before ID 20)        [ PASS ]
...
SUMMARY: Passed: X | Failed: 0
```


## 🔍 Key Features

- **Memory Safe**: Proper destructors and memory management
- **Edge Case Handling**: Comprehensive validation for empty inputs and boundary conditions
- **Modular Design**: Clean separation of concerns with abstract interfaces
- **Performance Optimized**: Space-optimized DP solutions, efficient graph algorithms
- **Well-Documented**: Clear comments explaining algorithm choices and complexities

## 📝 Implementation Highlights

### Double Hashing
- Uses two hash functions to minimize clustering
- Implements lazy deletion for efficient removal

### Skip List
- Probabilistic balancing with random level generation
- Maintains sorted order for efficient range queries

### Red-Black Tree
- Self-balancing with O(log n) guaranteed operations
- Implements both left and right rotations
- Proper fixup procedures after insertion and deletion

### Dynamic Programming
- Space-optimized solutions using 1D arrays where possible
- Handles large numbers with modulo arithmetic

### Graph Algorithms
- Union-Find with path compression and union by rank
- BFS for unweighted shortest paths
- Floyd-Warshall for dense graphs

