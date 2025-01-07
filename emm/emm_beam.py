from collections import deque
from .priority_queue import PriorityQueue
from .constraints import satisfies_all


def emm_beam(dataset, quality_measure, refinement_operator, w, d, q, constraints):

    # initialize candidate queue
    candidate_queue = deque()

    # start with empty description
    candidate_queue.append([])

    # initialize result set with max size q (global best)
    result_set = PriorityQueue(max_size=q)

    # for each level until maximum depth
    for level in range(1, d + 1):

        # initialize beam with max size w (local best)
        beam = PriorityQueue(max_size=w)

        # while candidate queue is not empty
        while candidate_queue:

            # get best description (a list of descriptions),
            # in our case the description with the lowest average survival probability
            seed = candidate_queue.popleft()

            # generate new descriptions by applying refinement operator
            new_desc = refinement_operator(seed)

            # for each new description
            for desc in new_desc:

                # calculate the quality measure
                quality = quality_measure(desc)

                # if the description satisfies all constraints
                if satisfies_all(desc, dataset, constraints):

                    # insert the description into result set
                    result_set.insert_with_priority(desc, quality)

                    # insert the description into beam
                    beam.insert_with_priority(desc, quality)

        # while beam is not empty
        while beam:

            # get the best description and remove it
            best_desc = beam.get_pop_front_element()

            # # remove the best description from beam
            # beam.heap.pop(0)

            # update the best description to candidate queue
            # to be considered in the next level
            candidate_queue.append(best_desc)

    # return the result set which has the best descriptions
    return result_set
