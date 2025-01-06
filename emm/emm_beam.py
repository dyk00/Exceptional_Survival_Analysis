from collections import deque
from .priority_queue import PriorityQueue


def emm_beam(dataset, quality_measure, refinement_operator, w, d, q, constraints):

    candidate_queue = deque()
    candidate_queue.append([])
    result_set = PriorityQueue(max_size=q)

    for level in range(1, d + 1):
        beam = PriorityQueue(max_size=w)

        while candidate_queue:
            seed = candidate_queue.popleft()
            new_desc = refinement_operator(seed)

            for desc in new_desc:
                quality = quality_measure(desc)

                if satisfies_all(constraints):
                    result_set.insert_with_priority(desc, quality)
                    beam.insert_with_priority(desc, quality)

        while not beam.empty():
            best_desc = beam.get_front_element()
            beam.heap.pop(0)
            candidate_queue.append(best_desc)

    return result_set
