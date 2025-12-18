// ArcadiaEngine.cpp - STUDENT TEMPLATE
// TODO: Implement all the functions below according to the assignment requirements

#include "ArcadiaEngine.h"
#include <algorithm>
#include <queue>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>

using namespace std;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

// --- 1. PlayerTable (Double Hashing) ---

class ConcretePlayerTable : public PlayerTable {
private:
    static const int TABLE_SIZE = 101;

    struct Entry {
        int playerID;
        string name;
        bool occupied;
        bool deleted;  // For lazy deletion

        Entry() : playerID(-1), name(""), occupied(false), deleted(false) {}
    };
    vector<Entry> table;
    // Primary hash function
    int h1(int key) {
        return key % TABLE_SIZE;
    }

    // Secondary hash function (must return non-zero value)
    int h2(int key) {
        return 1 + (key % (TABLE_SIZE - 1));
    }

    // Double hashing function: h(key, i) = (h1(key) + i * h2(key)) % TABLE_SIZE
    int hash(int key, int i) {
        return (h1(key) + i * h2(key)) % TABLE_SIZE;
    }
public:
    ConcretePlayerTable() {
        table.resize(TABLE_SIZE);
    }

    void insert(int playerID, string name) override {
        // Try to find an empty slot using double hashing
        for (int i = 0; i < TABLE_SIZE; i++) {
            int index = hash(playerID, i);

            // If slot is empty or marked as deleted, we can insert here
            if (!table[index].occupied || table[index].deleted) {
                table[index].playerID = playerID;
                table[index].name = name;
                table[index].occupied = true;
                table[index].deleted = false;
                return;
            }

            // If the playerID already exists, update the name
            if (table[index].playerID == playerID) {
                table[index].name = name;
                return;
            }
        }

        // If we've probed all slots and found no space, table is full
        throw runtime_error("Table is full");
    }

    string search(int playerID) override {
        // Use double hashing to search for the player
        for (int i = 0; i < TABLE_SIZE; i++) {
            int index = hash(playerID, i);

            // If we hit an empty slot that was never occupied, player doesn't exist
            if (!table[index].occupied) {
                return "";
            }

            // If we find the playerID and it's not deleted
            if (table[index].occupied && !table[index].deleted &&
                table[index].playerID == playerID) {
                return table[index].name;
            }
        }

        // Player not found after probing all slots
        return "";
    }
};

// --- 2. Leaderboard (Skip List) ---

class ConcreteLeaderboard : public Leaderboard {
private:
    static const int MAX_LEVEL = 16;
    
    struct Node {
        int playerID;
        int score;
        vector<Node*> forward;  // Forward pointers for each level
        
        Node(int id, int sc, int level) : playerID(id), score(sc) {
            forward.resize(level + 1, nullptr);
        }
    };
    
    Node* head;
    int maxLevel;  // Current maximum level in the skip list
    
    // Random level generator for skip list
    int randomLevel() {
        int level = 0;
        while (rand() % 2 == 0 && level < MAX_LEVEL - 1) {
            level++;
        }
        return level;
    }
    
    // Comparison function: Returns true if node1 should come BEFORE node2
    // Order: Higher score first, then lower ID for ties
    bool shouldComeBefore(int score1, int id1, int score2, int id2) {
        if (score1 != score2) {
            return score1 > score2;  // Higher score comes first (descending)
        }
        return id1 < id2;  // For ties, lower ID comes first (ascending)
    }

public:
    ConcreteLeaderboard() {
        maxLevel = 0;
        // Create head node with dummy values (will always be smallest)
        head = new Node(-1, -1, MAX_LEVEL);
    }
    
    ~ConcreteLeaderboard() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->forward[0];
            delete current;
            current = next;
        }
    }

    void addScore(int playerID, int score) override {
        vector<Node*> update(MAX_LEVEL, nullptr);
        Node* current = head;
        
        // Find the insertion position and track update pointers
        for (int i = maxLevel; i >= 0; i--) {
            while (current->forward[i] != nullptr &&
                   shouldComeBefore(current->forward[i]->score, 
                                   current->forward[i]->playerID,
                                   score, playerID)) {
                current = current->forward[i];
            }
            update[i] = current;
        }
        
        // Move to the next node (potential duplicate position)
        current = current->forward[0];
        
        // Check if player already exists - if so, remove old entry first
        if (current != nullptr && current->playerID == playerID) {
            removePlayer(playerID);
            // Re-find insertion position after removal
            current = head;
            for (int i = maxLevel; i >= 0; i--) {
                while (current->forward[i] != nullptr &&
                       shouldComeBefore(current->forward[i]->score, 
                                       current->forward[i]->playerID,
                                       score, playerID)) {
                    current = current->forward[i];
                }
                update[i] = current;
            }
        }
        
        // Generate random level for new node
        int newLevel = randomLevel();
        
        // If new level is greater than current max level, update head pointers
        if (newLevel > maxLevel) {
            for (int i = maxLevel + 1; i <= newLevel; i++) {
                update[i] = head;
            }
            maxLevel = newLevel;
        }
        
        // Create new node and insert it
        Node* newNode = new Node(playerID, score, newLevel);
        for (int i = 0; i <= newLevel; i++) {
            newNode->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = newNode;
        }
    }

    void removePlayer(int playerID) override {
        vector<Node*> update(MAX_LEVEL, nullptr);
        Node* current = head;
        
        // Find the node to delete using linear scan at level 0
        // Track update pointers at all levels
        for (int i = maxLevel; i >= 0; i--) {
            while (current->forward[i] != nullptr &&
                   current->forward[i]->playerID != playerID &&
                   (shouldComeBefore(current->forward[i]->score,
                                    current->forward[i]->playerID,
                                    INT_MIN, playerID) ||
                    current->forward[i]->playerID < playerID)) {
                current = current->forward[i];
            }
            update[i] = current;
        }
        
        // Get the node to delete
        current = current->forward[0];
        
        // If node found, delete it
        if (current != nullptr && current->playerID == playerID) {
            // Update forward pointers at all levels
            for (int i = 0; i <= maxLevel; i++) {
                if (update[i]->forward[i] != current) {
                    break;
                }
                update[i]->forward[i] = current->forward[i];
            }
            
            delete current;
            
            // Update max level if necessary
            while (maxLevel > 0 && head->forward[maxLevel] == nullptr) {
                maxLevel--;
            }
        }
    }

    vector<int> getTopN(int n) override {
        vector<int> result;
        Node* current = head->forward[0];  // Start from first real node
        
        // Traverse the bottom level to get top N players
        int count = 0;
        while (current != nullptr && count < n) {
            result.push_back(current->playerID);
            current = current->forward[0];
            count++;
        }
        
        return result;
    }
};
// --- 3. AuctionTree (Red-Black Tree) ---

class ConcreteAuctionTree : public AuctionTree {
private:
      enum Color { RED, BLACK };
    
    struct Node {
        int itemID;
        int price;
        Color color;
        Node* left;
        Node* right;
        Node* parent;
        
        Node(int id, int p) : itemID(id), price(p), color(RED), 
                               left(nullptr), right(nullptr), parent(nullptr) {}
    };
    
    Node* root;
    Node* NIL;  // Sentinel node for leaf nodes
    
    // Helper: Compare nodes (price ascending, then ID ascending for ties)
    bool isLessThan(Node* a, int priceB, int idB) {
        if (a->price != priceB) {
            return a->price < priceB;
        }
        return a->itemID < idB;
    }
    
    bool isLessThan(Node* a, Node* b) {
        return isLessThan(a, b->price, b->itemID);
    }
    
    // Rotations
    void leftRotate(Node* x) {
        Node* y = x->right;
        x->right = y->left;
        
        if (y->left != NIL) {
            y->left->parent = x;
        }
        
        y->parent = x->parent;
        
        if (x->parent == nullptr) {
            root = y;
        } else if (x == x->parent->left) {
            x->parent->left = y;
        } else {
            x->parent->right = y;
        }
        
        y->left = x;
        x->parent = y;
    }
    
    void rightRotate(Node* x) {
        Node* y = x->left;
        x->left = y->right;
        
        if (y->right != NIL) {
            y->right->parent = x;
        }
        
        y->parent = x->parent;
        
        if (x->parent == nullptr) {
            root = y;
        } else if (x == x->parent->right) {
            x->parent->right = y;
        } else {
            x->parent->left = y;
        }
        
        y->right = x;
        x->parent = y;
    }
    
    // Insert fixup
    void insertFixup(Node* z) {
        while (z->parent != nullptr && z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) {
                Node* y = z->parent->parent->right;  // Uncle
                
                if (y->color == RED) {
                    // Case 1: Uncle is red
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->right) {
                        // Case 2: z is right child
                        z = z->parent;
                        leftRotate(z);
                    }
                    // Case 3: z is left child
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    rightRotate(z->parent->parent);
                }
            } else {
                Node* y = z->parent->parent->left;  // Uncle
                
                if (y->color == RED) {
                    // Case 1: Uncle is red
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->left) {
                        // Case 2: z is left child
                        z = z->parent;
                        rightRotate(z);
                    }
                    // Case 3: z is right child
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    leftRotate(z->parent->parent);
                }
            }
        }
        root->color = BLACK;
    }
    
    // Transplant helper for deletion
    void transplant(Node* u, Node* v) {
        if (u->parent == nullptr) {
            root = v;
        } else if (u == u->parent->left) {
            u->parent->left = v;
        } else {
            u->parent->right = v;
        }
        v->parent = u->parent;
    }
    
    // Find minimum node in subtree
    Node* minimum(Node* node) {
        while (node->left != NIL) {
            node = node->left;
        }
        return node;
    }
    
    // Delete fixup
    void deleteFixup(Node* x) {
        while (x != root && x->color == BLACK) {
            if (x == x->parent->left) {
                Node* w = x->parent->right;  // Sibling
                
                if (w->color == RED) {
                    // Case 1: Sibling is red
                    w->color = BLACK;
                    x->parent->color = RED;
                    leftRotate(x->parent);
                    w = x->parent->right;
                }
                
                if (w->left->color == BLACK && w->right->color == BLACK) {
                    // Case 2: Sibling's children are both black
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->right->color == BLACK) {
                        // Case 3: Sibling's right child is black
                        w->left->color = BLACK;
                        w->color = RED;
                        rightRotate(w);
                        w = x->parent->right;
                    }
                    // Case 4: Sibling's right child is red
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->right->color = BLACK;
                    leftRotate(x->parent);
                    x = root;
                }
            } else {
                Node* w = x->parent->left;  // Sibling
                
                if (w->color == RED) {
                    // Case 1: Sibling is red
                    w->color = BLACK;
                    x->parent->color = RED;
                    rightRotate(x->parent);
                    w = x->parent->left;
                }
                
                if (w->right->color == BLACK && w->left->color == BLACK) {
                    // Case 2: Sibling's children are both black
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->left->color == BLACK) {
                        // Case 3: Sibling's left child is black
                        w->right->color = BLACK;
                        w->color = RED;
                        leftRotate(w);
                        w = x->parent->left;
                    }
                    // Case 4: Sibling's left child is red
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->left->color = BLACK;
                    rightRotate(x->parent);
                    x = root;
                }
            }
        }
        x->color = BLACK;
    }
    
    // O(N) search for node by itemID
    Node* findByID(Node* node, int itemID) {
        if (node == NIL) {
            return nullptr;
        }
        
        if (node->itemID == itemID) {
            return node;
        }
        
        Node* leftResult = findByID(node->left, itemID);
        if (leftResult != nullptr) {
            return leftResult;
        }
        
        return findByID(node->right, itemID);
    }
    
    // Cleanup helper
    void destroyTree(Node* node) {
        if (node == NIL) {
            return;
        }
        destroyTree(node->left);
        destroyTree(node->right);
        delete node;
    }
public:
    ConcreteAuctionTree() {
        NIL = new Node(0, 0);
        NIL->color = BLACK;
        NIL->left = nullptr;
        NIL->right = nullptr;
        NIL->parent = nullptr;
        root = NIL;
    }
    
    ~ConcreteAuctionTree() {
        destroyTree(root);
        delete NIL;
    }

    void insertItem(int itemID, int price) override {
        Node* z = new Node(itemID, price);
        z->left = NIL;
        z->right = NIL;
        
        Node* y = nullptr;
        Node* x = root;
        
        // Standard BST insertion with composite key comparison
        while (x != NIL) {
            y = x;
            if (isLessThan(z, x)) {
                x = x->left;
            } else {
                x = x->right;
            }
        }
        
        z->parent = y;
        
        if (y == nullptr) {
            root = z;
        } else if (isLessThan(z, y)) {
            y->left = z;
        } else {
            y->right = z;
        }
        
        // New node starts as red
        z->color = RED;
        
        // Fix Red-Black properties
        insertFixup(z);
    }

    void deleteItem(int itemID) override {
        // O(N) search for the node by itemID
        Node* z = findByID(root, itemID);
        
        if (z == nullptr) {
            return;  // Item not found
        }
        
        // Standard Red-Black Tree deletion
        Node* y = z;
        Node* x;
        Color yOriginalColor = y->color;
        
        if (z->left == NIL) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == NIL) {
            x = z->left;
            transplant(z, z->left);
        } else {
            y = minimum(z->right);
            yOriginalColor = y->color;
            x = y->right;
            
            if (y->parent == z) {
                x->parent = y;
            } else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }
            
            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }
        
        delete z;
        
        // Fix Red-Black properties if needed
        if (yOriginalColor == BLACK) {
            deleteFixup(x);
        }
    }
};
// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================

int InventorySystem::optimizeLootSplit(int n, vector<int>& coins) {
    // TODO: Implement partition problem using DP
    // Goal: Minimize |sum(subset1) - sum(subset2)|
    // Hint: Use subset sum DP to find closest sum to total/2
    return 0;
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items) {
    // TODO: Implement 0/1 Knapsack using DP
    // items = {weight, value} pairs
    // Return maximum value achievable within capacity
    return 0;
}

long long InventorySystem::countStringPossibilities(string s) {
    // TODO: Implement string decoding DP
    // Rules: "uu" can be decoded as "w" or "uu"
    //        "nn" can be decoded as "m" or "nn"
    // Count total possible decodings
    return 0;
}

// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================

bool WorldNavigator::pathExists(int n, vector<vector<int>>& edges, int source, int dest) {
    // TODO: Implement path existence check using BFS or DFS
    // edges are bidirectional
    return false;
}

long long WorldNavigator::minBribeCost(int n, int m, long long goldRate, long long silverRate,
                                       vector<vector<int>>& roadData) {
    // TODO: Implement Minimum Spanning Tree (Kruskal's or Prim's)
    // roadData[i] = {u, v, goldCost, silverCost}
    // Total cost = goldCost * goldRate + silverCost * silverRate
    // Return -1 if graph cannot be fully connected
    return -1;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
    // TODO: Implement All-Pairs Shortest Path (Floyd-Warshall)
    // Sum all shortest distances between unique pairs (i < j)
    // Return the sum as a binary string
    // Hint: Handle large numbers carefully
    return "0";
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char>& tasks, int n) {
    // TODO: Implement task scheduler with cooling time
    // Same task must wait 'n' intervals before running again
    // Return minimum total intervals needed (including idle time)
    // Hint: Use greedy approach with frequency counting
    return 0;
}

// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C" {
    PlayerTable* createPlayerTable() { 
        return new ConcretePlayerTable(); 
    }

    Leaderboard* createLeaderboard() { 
        return new ConcreteLeaderboard(); 
    }

    AuctionTree* createAuctionTree() { 
        return new ConcreteAuctionTree(); 
    }
}
