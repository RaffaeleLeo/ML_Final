class AttNode:
    def __init__(self, attribute):
        self.attribute = attribute
        self.leaves = []

    def add_leaf(self, node):
        self.leaves.append(node)