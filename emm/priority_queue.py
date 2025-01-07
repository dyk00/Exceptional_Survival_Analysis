import heapq


class PriorityQueue:
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.heap = []

    # insert a tuple (priority, description) into min-heap
    def insert_with_priority(self, description, priority):
        tuple = (priority, description)
        heapq.heappush(self.heap, tuple)

        # if exceeds the max size, remove description with largest priority
        # (i.e., quality measure)
        if self.max_size is not None and len(self.heap) > self.max_size:

            # single description
            largest = heapq.nlargest(1, self.heap)[0]
            self.heap.remove(largest)

            # reheapify
            heapq.heapify(self.heap)

    # return True if the heap is empty
    def empty(self):
        return len(self.heap) == 0

    # # return the description with smallest priority
    # def get_front_element(self):
    #     if self.empty():
    #         return None
    #     # [][]: [order of heap], [priority/description]
    #     return self.heap[0][1]

    # return and remove the description with smallest priority
    def get_pop_front_element(self):
        if self.empty():
            return None
        priority, description = heapq.heappop(self.heap)
        return description
