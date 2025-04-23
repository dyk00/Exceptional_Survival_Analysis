from collections import deque
from .priority_queue import PriorityQueue
from .constraints import satisfies_all
from .refinement import refinement_operator
from .quality_measure import quality_measure
from .description import description_to_string
import heapq


def emm_beam(dataset, columns, w, d, q, constraints, b=2):

    # initialize candidate queue
    # double-ended queue with O(1) performance for adding and removing
    candidate_queue = deque()

    # start with empty description
    candidate_queue.append([])

    # initialize result set with max size q (global best)
    result_set = PriorityQueue(max_size=q)

    # for each level until maximum depth
    for level in range(1, d + 1):

        print(f"\n== Depth {level} ==")

        # initialize beam with max size w (local best)
        beam = PriorityQueue(max_size=w)

        # while candidate queue is not empty
        while candidate_queue:

            # get one description at a time in BFS manner per depth
            seed_desc = candidate_queue.popleft()

            # generate new descriptions by applying refinement operator
            # level for printing splits
            new_desc = refinement_operator(seed_desc, dataset, columns, b, level)

            # for each new description
            for desc in new_desc:

                # calculate the quality measure
                quality = quality_measure(desc, dataset)

                # if the description satisfies all constraints
                if satisfies_all(desc, dataset, constraints):

                    # insert the description into result set
                    result_set.insert_with_priority(desc, quality)

                    # insert the description into beam
                    beam.insert_with_priority(desc, quality)

        #         with open("results.txt", "a") as f:
        #             for desc in new_desc:
        #                 quality = quality_measure(desc, dataset)
        #                 satisfies = satisfies_all(desc, dataset, constraints)
        #                 desc_str = description_to_string(desc)
        #                 f.write(f"Description: {desc_str} | Quality: {quality} | Satisfies constraints: {satisfies}\n")

        # while beam is not empty
        while not beam.empty():

            # get candidate descriptions and remove them from beam one by one,
            candidate_desc = beam.get_pop_front_element()

            # update the descriptions to candidate queue
            # to be considered in the next level
            if candidate_desc is not None:
                candidate_queue.append(list(candidate_desc))

    # return the result set which has the best descriptions
    return result_set
