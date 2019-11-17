class node:
    def __init__(self, val, prev = None):
        self.val = val
        self.prev = prev
        self.next = None

class linkedList:
    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0
        self._hashMap = {}

    def __getitem__(self, key):
        return self._hashMap[key]

    def push_back(self, value):
        if not self.contains(value):
            new_node = node(value, self._tail)
            if not self.empty():
                self._tail.next = new_node
            else:
                self._head = new_node
            self._tail = new_node
            self._hashMap[value] = new_node
            self._size += 1
        else:
            print("Value is already used")

    def push_front(self, value):
        if not self.contains(value):
            new_node = node(value)
            new_node.next = self._head
            if not self.empty():
                self._head.prev = new_node
            self._head = new_node
            self._hashMap[value] = new_node
            self._size += 1
        else:
            print("Value has already been used!")

    def remove(self, value):
        node = self.find(value)
        if node is not None:
            if node.prev is not None:
                node.prev.next = node.next
            if node.next is not None:
                node.next.prev = node.prev
            self._hashMap.pop(value)
            self._size -= 1
        else:
            print("Node not found!")

    def find(self, value):
        if self.contains(value):
            return self._hashMap[value]
        else:
            return None

    def contains(self, value):
        return value in self._hashMap

    def print(self):
        node = self._head
        for index in range(0, self._size):
            if index is 0:
                print(str(node.value), end = " ")
            else:
                print(" -> " + str(node.value), end = " ")
            node = node.next
        print("\nEnd of list!")

    def reversePrint(self):
        node = self._tail
        for index in range(0, self._size):
            if index is 0:
                print(str(node.value), end = " ")
            else:
                print(" -> " + str(node.value), end = " ")
            node = node.prev
        print("\nEnd of reverse list!")

    def getSize(self):
        return self._size

    def empty(self):
        return self.getSize() == 0
