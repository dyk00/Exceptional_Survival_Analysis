import heapq
from .description import sort_description


class PriorityQueue:
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.heap = []

        # maintain a set for removing duplicate
        self.seen = set()

    # insert a tuple (-priority, description) into min-heap
    # negative priority is easy to handle
    # when removing description with actual highest probability
    def insert_with_priority(self, description, quality):
        priority = -quality
        sorted_desc = tuple(sort_description(description))

        # if description is not in seen, push to heap and seen
        if sorted_desc not in self.seen:
            heapq.heappush(self.heap, (priority, sorted_desc))
            self.seen.add(sorted_desc)

            # if exceeds the max size, remove description with acutal highest probability
            # (i.e., quality measure)
            # as we're focusing on finding descriptions with lowest probability
            if self.max_size is not None and len(self.heap) > self.max_size:
                removed = heapq.heappop(self.heap)
                self.seen.remove(removed[1])

    # return True if the heap is empty
    def empty(self):
        return len(self.heap) == 0

    # return and remove the description with smallest priority
    def get_pop_front_element(self):
        if self.empty():
            return None
        priority, description = heapq.heappop(self.heap)
        self.seen.remove(description)
        return description

    # making heap iterable for printing in main
    def __iter__(self):
        return iter(self.heap)
