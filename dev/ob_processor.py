from copy import deepcopy
import numpy as np
#import logging
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)


MAX_NUM_OBSTACLE = 3
NORMALIZE_INDICE = np.asarray([1, 18, 22, 24, 26, 28, 30, 32, 34])


class ObservationProcessor:

    def process(self, ob):
        return ob

    def get_aug_dim(self):
        return 0

    def reset(self):
        pass


class BodySpeedAugmentor(ObservationProcessor):

    def __init__(self):
        self.last_observation = None
        self.num_body_parts = 14
        self.body_parts_ix = np.arange(22, 36)
        self.zero_padding = [0]*self.num_body_parts
        self.obstacle_pos = set()

    def process(self, ob):

        # deal with obstacles
        pelvis_x = ob[1]
        ob_x = ob[-3] + pelvis_x
        self.obstacle_pos.add(ob_x)
        if len(self.obstacle_pos) >= MAX_NUM_OBSTACLE:
            ob[-3:] = [0]*3

        # generate velocity for body parts
        cur_observation = np.asarray(ob)[self.body_parts_ix]
        if self.last_observation is None:
            res = ob + self.zero_padding
        else:
            ob_augmentation = cur_observation - self.last_observation
            res = ob + ob_augmentation.tolist()
        self.last_observation = cur_observation

        # normalize x
        res = np.asarray(res)
        res[NORMALIZE_INDICE] -= res[NORMALIZE_INDICE].min()
        res = res.tolist()

#        logger.info("observation:")
#        for i, x in enumerate(res):
#            logger.info("{}: {}".format(i+1, x))
#        logger.info("-"*50)

        return res

    def get_aug_dim(self):
        return self.num_body_parts

    def reset(self):
#        logger.info("ob_processor.reset()")
        self.last_observation = None
        self.obstacle_pos.clear()


class SecondOrderAugmentor(ObservationProcessor):

    def __init__(self):

        # last observations in the order
        # 1. pelvis velocity
        # 2. ankle, knee and hip velocity
        # 3. center of mass velocity
        # 4. body position
        self.last_observations = [None, None, None, None]

        # indices for each part
        # 1. pelvis velocity: 3-5
        # 2. ankle, knee and hip velocity: 12-17
        # 3. center of mass velocity: 20, 21
        # 4. body position: 22-35
        self.ix = [np.arange(3, 6), np.arange(12, 18), np.arange(20, 22), np.arange(22, 36)]

        self.aug_len = 0
        for ix in self.ix:
            self.aug_len += ix.size
        self.zero_padding = [0]*self.aug_len

        self.obstacle_pos = set()

    def get_aug_dim(self):
        return self.aug_len

    def process(self, ob):

        # deal with obstacles
        pelvis_x = ob[1]
        ob_x = ob[-3] + pelvis_x
        self.obstacle_pos.add(ob_x)
        if len(self.obstacle_pos) >= MAX_NUM_OBSTACLE:
            ob[-3:] = [0]*3

        # generate velocities
        cur_observations = [np.asarray(ob)[ix] for ix in self.ix]
        res = deepcopy(ob)
        for i, last_observation in enumerate(self.last_observations):
            if last_observation is None:
                res += [0]*self.ix[i].size
            else:
                ob_augmentation = cur_observations[i] - last_observation
                res += ob_augmentation.tolist()
        self.last_observations = cur_observations

        # normalize x
        res = np.asarray(res)
        res[NORMALIZE_INDICE] -= res[NORMALIZE_INDICE].min()
        res = res.tolist()

        return res

    def reset(self):
#        logger.info("ob_processor.reset()")
        self.last_observations = [None, None, None, None]
        self.obstacle_pos.clear()
