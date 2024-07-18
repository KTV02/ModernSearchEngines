from flask import jsonify
class TreeNode:
    def __init__(self, word, count, left=None, right=None):
        self.word = word
        self.count = count
        self.left = left
        self.right = right

def create_sample_tree():
    # Creating a simple binary tree for demonstration
    return TreeNode(
        "root", 10,
        left=TreeNode(
            "left", 5,
            left=TreeNode("left.left", 2,
                          left=TreeNode("left.left.left", 2),
                          right= TreeNode("left.left.right", 2)),
            right=TreeNode("left.right", 3)
        ),
        right=TreeNode(
            "right", 15,
            left=TreeNode("right.left", 12),
            right=TreeNode("right.right", 18)
        )
    )

def tree_to_dict(node):
    if not node:
        return None
    return {
        "word": node.word,
        "count": node.count,
        "left": tree_to_dict(node.left),
        "right": tree_to_dict(node.right)
    }

def get_tree():
    print("Getting tree...")
    tree = create_sample_tree()
    return jsonify(tree_to_dict(tree))