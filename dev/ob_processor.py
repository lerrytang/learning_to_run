from copy import deepcopy
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
The observation contains 41 values:

position of the pelvis (rotation, x, y)
velocity of the pelvis (rotation, x, y)
rotation of each ankle, knee and hip (6 values)
angular velocity of each ankle, knee and hip (6 values)
position of the center of mass (2 values)
velocity of the center of mass (2 values)
positions (x, y) of head, pelvis, torso, left and right toes, left and right talus (14 values)
strength of left and right psoas: 1 for difficulty < 2, otherwise a random normal variable with mean 1 and standard deviation 0.1 fixed for the entire simulation
next obstacle: x distance from the pelvis, y position of the center relative to the the ground, radius.
"""

MAX_NUM_OBSTACLE = 3

PELVIS_X_IX = 1
PELVIS_Y_IX = 2
X_NORMALIZE_INDICE = np.asarray([1, 18, 22, 24, 26, 28, 30, 32, 34])
Y_NORMALIZE_INDICE = X_NORMALIZE_INDICE[1:] + 1    # reserve absolute value of y of pelvis

PELVIS_VEL_X_IX = 4
PELVIS_VEL_Y_IX = 5
VEL_X_NORMALIZE_INDICE = np.asarray([4, 20] + [41, 43, 45, 47, 49, 51, 53])
VEL_Y_NORMALIZE_INDICE = VEL_X_NORMALIZE_INDICE + 1

BODY_PARTS_IX = np.arange(22, 36)

PSOAS_IX = np.asarray([36, 37])

OBSTACLE_X_IX = 38
OBSTACLE_IX = np.asarray([38, 39, 40])


class ObservationProcessor:

    def process(self, ob):
        return ob

    def get_aug_dim(self):
        return 0

    def reset(self):
        pass


class NormalizedFirstOrder(ObservationProcessor):

    def __init__(self):
        self.last_observation = None
        self.obstacle_pos = set()

    def process(self, ob):

        # generate velocity for body parts
        if self.last_observation is None:
            res = ob + [0] * BODY_PARTS_IX.size
        else:
            # times 100 because each step is 0.01sec
            ob_augmentation = (np.asarray(ob) - self.last_observation) * 100.0
            res = ob + ob_augmentation[BODY_PARTS_IX].tolist()
        self.last_observation = np.asarray(ob)

        # normalize
        res = np.asarray(res)
        res[X_NORMALIZE_INDICE] -= res[PELVIS_X_IX]
        res[Y_NORMALIZE_INDICE] -= res[PELVIS_Y_IX]
        res[VEL_X_NORMALIZE_INDICE] -= res[PELVIS_VEL_X_IX]
        res[VEL_Y_NORMALIZE_INDICE] -= res[PELVIS_VEL_Y_IX]
        res[PSOAS_IX] -= 1.0

        # deal with obstacles
        if len(self.obstacle_pos) < MAX_NUM_OBSTACLE:
            ob_x = round(res[OBSTACLE_X_IX] + ob[PELVIS_X_IX], 4)
            self.obstacle_pos.add(ob_x)
        else:
            # set invisible size
            res[OBSTACLE_X_IX] = 0

        return res.tolist()

    def get_aug_dim(self):
        return BODY_PARTS_IX.size

    def reset(self):
        self.last_observation = None
        self.obstacle_pos.clear()


class NormalizedSecondOrder(ObservationProcessor):

    def __init__(self):
        self.last_observation = None
        self.last_velocity = None
        self.obstacle_pos = set()

    def process(self, ob):

        # generate velocity for body parts
        current_ob = np.asarray(ob)
        if self.last_observation is None:
            res = ob + [0] * BODY_PARTS_IX.size
            current_vel = None
        else:
            # times 100 because each step is 0.01sec
            _1st_order_diff = (current_ob - self.last_observation) * 100.0
            res = ob + _1st_order_diff[BODY_PARTS_IX].tolist()
            current_vel = np.asarray(res)[VEL_X_NORMALIZE_INDICE]
        self.last_observation = current_ob

        # generate accelerations
        if current_vel is None or self.last_velocity is None:
            res += [0] * VEL_X_NORMALIZE_INDICE.size
        else:
            # times 100 because each step is 0.01sec
            _2nd_order_diff = (current_vel - self.last_velocity) * 100.0
            res += _2nd_order_diff.tolist()
        self.last_velocity = current_vel

        # normalize
        res = np.asarray(res)

        # logger.info("Observation: {}".format(res[:41]))
        # logger.info("body_parts: {}".format(res[BODY_PARTS_IX]))
        # logger.info("Velocities: {}".format(res[VEL_X_NORMALIZE_INDICE]))
        # logger.info("Accelerations: {}".format(res[-VEL_X_NORMALIZE_INDICE.size:]))
        # logger.info("-"*50)

        # velocities
        res[X_NORMALIZE_INDICE] -= res[PELVIS_X_IX]
        res[Y_NORMALIZE_INDICE] -= res[PELVIS_Y_IX]
        res[VEL_X_NORMALIZE_INDICE] -= res[PELVIS_VEL_X_IX]
        res[VEL_Y_NORMALIZE_INDICE] -= res[PELVIS_VEL_Y_IX]

        # # accelerations
        # res[-VEL_X_NORMALIZE_INDICE.size:] -= res[-VEL_X_NORMALIZE_INDICE.size]

        res[PSOAS_IX] -= 1.0

        # deal with obstacles
        if len(self.obstacle_pos) < MAX_NUM_OBSTACLE:
            ob_x = round(res[OBSTACLE_X_IX] + ob[PELVIS_X_IX], 4)
            self.obstacle_pos.add(ob_x)
        else:
            # set invisible size
            res[OBSTACLE_X_IX] = 0

        return res.tolist()

    def get_aug_dim(self):
        return BODY_PARTS_IX.size + VEL_X_NORMALIZE_INDICE.size

    def reset(self):
        self.last_observation = None
        self.last_velocity = None
        self.obstacle_pos.clear()


class BodySpeedAugmentor(ObservationProcessor):

    def __init__(self):
        self.last_observation = None
        self.num_body_parts = 14
        self.body_parts_ix = np.arange(22, 36)
        self.zero_padding = [0]*self.num_body_parts
        self.obstacle_pos = set()

    def process(self, ob):

        # deal with obstacles
        if len(self.obstacle_pos) < MAX_NUM_OBSTACLE:
            pelvis_x = ob[PELVIS_X_IX]
            ob_x = ob[-3] + pelvis_x
            self.obstacle_pos.add(ob_x)
        else:
            ob[-3:] = [0]*3

        # generate velocity for body parts
        cur_observation = np.asarray(ob)[self.body_parts_ix]
        if self.last_observation is None:
            res = ob + self.zero_padding
        else:
            # TODO: this is a bug (need to time 100)
            # but because some trials are trained on this, I leave it as-is
            ob_augmentation = cur_observation - self.last_observation
            res = ob + ob_augmentation.tolist()
        self.last_observation = cur_observation

        # normalize x
        res = np.asarray(res)
#        res[NORMALIZE_INDICE] -= res[PELVIS_IX]
        res[X_NORMALIZE_INDICE] -= res[X_NORMALIZE_INDICE].min()
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
        if len(self.obstacle_pos) < MAX_NUM_OBSTACLE:
            pelvis_x = ob[PELVIS_X_IX]
            ob_x = ob[-3] + pelvis_x
            self.obstacle_pos.add(ob_x)
        else:
            ob[-3:] = [0]*3

        # generate velocities
        cur_observations = [np.asarray(ob)[ix] for ix in self.ix]
        res = deepcopy(ob)
        for i, last_observation in enumerate(self.last_observations):
            if last_observation is None:
                res += [0]*self.ix[i].size
            else:
                # TODO: this is a bug (need to time 100)
                # but because some trials are trained on this, I leave it as-is
                ob_augmentation = cur_observations[i] - last_observation
                res += ob_augmentation.tolist()
        self.last_observations = cur_observations

        # normalize x
        res = np.asarray(res)
#        res[NORMALIZE_INDICE] -= res[PELVIS_IX]
        res[X_NORMALIZE_INDICE] -= res[X_NORMALIZE_INDICE].min()
        res = res.tolist()

        return res

    def reset(self):
#        logger.info("ob_processor.reset()")
        self.last_observations = [None, None, None, None]
        self.obstacle_pos.clear()
