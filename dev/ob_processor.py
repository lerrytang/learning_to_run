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

pos_rot_pelvis      0
pos_x_pelvis        1
pos_y_pelvis        2
vel_rot_pelvis      3
vel_x_pelvis        4
vel_y_pelvis        5

rot_hip_r           6
rot_knee_r          7
rot_ankle_r         8
rot_hip_l           9
rot_knee_l          10
rot_ankle_l         11
vel_hip_r           12
vel_knee_r          13
vel_ankle_r         14
vel_hip_l           15
vel_knee_l          16
vel_ankle_l         17

pos_x_mass          18
pos_y_mass          19
vel_x_mass          20
vel_y_mass          21

pos_x_head          22
pos_y_head          23
pos_x_pelvis        24   (redundant, same as 1)
pos_y_pelvis        25   (redundant, same as 2)
pos_x_torso         26
pos_y_torso         27
pos_x_toes_l        28
pos_y_toes_l        29
pos_x_toes_r        30
pos_y_toes_r        31
pos_x_talus_l       32
pos_y_talus_l       33
pos_x_talus_r       34
pos_y_talus_r       35

muscle_psoas_l      36
muscle_psoas_r      37

pos_x_obstacle      38
pos_y_obstacle      39
radius_obstacle     40
"""

MAX_NUM_OBSTACLE = 3

TOE_L_IX = 28
TOE_R_IX = 30

ORG_OB_DIM = 41
PELVIS_X_IX = 1
PELVIS_Y_IX = 2
X_NORMALIZE_INDICES = np.asarray([1, 18, 22, 24, 26, 28, 30, 32, 34])

BODY_PARTS_IX = np.arange(22, 36)
PSOAS_IX = np.asarray([36, 37])
OBSTACLE_X_IX = 38
OBSTACLE_IX = np.asarray([38, 39, 40])


def flip_observation(ob, to_negate, l_part, r_part):
    res = deepcopy(ob)
    res[:, to_negate] *= -1
    tmp = res[:, l_part]
    res[:, l_part] = res[:, r_part]
    res[:, r_part] = tmp
    return res


class ObservationProcessor(object):

    def process(self, ob):
        return ob

    def get_aug_dim(self):
        return 0

    def reset(self):
        pass

    def mirror_ob(self, ob):
        return None


class NormalizedFirstOrder(ObservationProcessor):
    """
    1. Append the diff of consecutive observations to the original one (all zeros if prev ob is None)
    2. If 3 obstacles are passed, set obstacle values to all zeros
    3. pos_x_XXX -= pos_x_pelvis
    """

    def __init__(self):
        self.last_observation = None
        self.obstacle_pos = set()

    def process(self, ob):

        # augment the original observation with diff
        if self.last_observation is None:
            res = ob + [0] * self.get_aug_dim()
        else:
            ob_augmentation = np.asarray(ob) - self.last_observation
            res = ob + ob_augmentation[:OBSTACLE_X_IX].tolist()
        self.last_observation = np.asarray(ob)
        res = np.asarray(res)

        # deal with obstacles
        if len(self.obstacle_pos) < MAX_NUM_OBSTACLE:
            ob_x = round(res[OBSTACLE_X_IX] + ob[PELVIS_X_IX], 4)
            self.obstacle_pos.add(ob_x)
        else:
            # set to invisible size and right beneath the pelvis
            res[OBSTACLE_IX] = 0

        # normalize
        res[X_NORMALIZE_INDICES] -= res[PELVIS_X_IX]
        res[PSOAS_IX] -= 1.0

        # logger.info(res)
        # logger.info("Distance between toes: {}".format(np.abs(res[28] - res[30])))
        # threshold = 0.01
        # if np.abs(res[28]-res[30]) >= threshold:
        #     logger.info("Distance between toes larger than {}".format(threshold))

        return res.tolist()

    def mirror_ob(self, ob0, action, reward, ob1, done, toe_dist_threshold):
        """
        Exchange left and right body parts (if it is pelvis rotation, negate)

        condition: ONLY IF the distance betwen the 2 toes are larger than toe_dist_threshold
        """

        # sanity check
        to_negate_org = np.asarray([0, 3])  # pelvis rotations
        to_negate_aug = to_negate_org + ORG_OB_DIM
        to_negate = np.append(to_negate_org, to_negate_aug)

        r_part_org = np.asarray([6, 7, 8, 12, 13, 14, 30, 31, 34, 35, 37])
        r_part_aug = r_part_org + ORG_OB_DIM
        r_part = np.append(r_part_org, r_part_aug)

        l_part_org = np.asarray([9, 10, 11, 15, 16, 17, 28, 29, 32, 33, 36])
        l_part_aug = l_part_org + ORG_OB_DIM
        l_part = np.append(l_part_org, l_part_aug)

        assert r_part.size == l_part.size
        assert np.intersect1d(l_part, r_part).size == 0

        # get indices of experiences that are qualified to mirror
        l_toe_x_pos = ob0[:, TOE_L_IX]
        r_toe_x_pos = ob0[:, TOE_R_IX]
        mask = np.abs(l_toe_x_pos - r_toe_x_pos) >= toe_dist_threshold

        if np.sum(mask) > 0:
            # augment ob0
            aug_ob0 = flip_observation(ob0[mask], to_negate, l_part, r_part)
            ob0 = np.concatenate([ob0, aug_ob0], axis=0)
            # augment ob1
            aug_ob1 = flip_observation(ob1[mask], to_negate, l_part, r_part)
            ob1 = np.concatenate([ob1, aug_ob1], axis=0)
            # augment action
            aug_action = deepcopy(action[mask])
            action = np.concatenate([action, aug_action], axis=0)
            # augment reward
            aug_reward = deepcopy(reward[mask])
            reward = np.concatenate([reward, aug_reward], axis=0)
            # augment done
            aug_done = deepcopy(done[mask])
            done = np.concatenate([done, aug_done], axis=0)

        return ob0, action, reward, ob1, done

    def get_aug_dim(self):
        return int(ORG_OB_DIM - OBSTACLE_IX.size)

    def reset(self):
        # logger.info("reset() called")
        self.last_observation = None
        self.obstacle_pos.clear()


# class NormalizedSecondOrder(ObservationProcessor):
#
#     def __init__(self):
#         self.last_observation = None
#         self.last_velocity = None
#         self.obstacle_pos = set()
#
#     def process(self, ob):
#
#         # generate velocity for body parts
#         current_ob = np.asarray(ob)
#         if self.last_observation is None:
#             res = ob + [0] * BODY_PARTS_IX.size
#             current_vel = None
#         else:
#             # times 100 because each step is 0.01sec
#             _1st_order_diff = (current_ob - self.last_observation) * 100.0
#             res = ob + _1st_order_diff[BODY_PARTS_IX].tolist()
#             current_vel = np.asarray(res)[VEL_X_NORMALIZE_INDICES]
#         self.last_observation = current_ob
#
#         # generate accelerations
#         if current_vel is None or self.last_velocity is None:
#             res += [0] * VEL_X_NORMALIZE_INDICES.size
#         else:
#             # times 100 because each step is 0.01sec
#             _2nd_order_diff = (current_vel - self.last_velocity) * 100.0
#             res += _2nd_order_diff.tolist()
#         self.last_velocity = current_vel
#
#         # normalize
#         res = np.asarray(res)
#
#         # deal with obstacles
#         if len(self.obstacle_pos) < MAX_NUM_OBSTACLE:
#             ob_x = round(res[OBSTACLE_X_IX] + ob[PELVIS_X_IX], 4)
#             self.obstacle_pos.add(ob_x)
#         else:
#             # set invisible size
#             res[OBSTACLE_X_IX] = 0
#
#         # logger.info("Observation: {}".format(res[:41]))
#         # logger.info("body_parts: {}".format(res[BODY_PARTS_IX]))
#         # logger.info("Velocities: {}".format(res[VEL_X_NORMALIZE_INDICE]))
#         # logger.info("Accelerations: {}".format(res[-VEL_X_NORMALIZE_INDICE.size:]))
#         # logger.info("-"*50)
#
#         # velocities
#         res[X_NORMALIZE_INDICE] -= res[PELVIS_X_IX]
#         res[Y_NORMALIZE_INDICE] -= res[PELVIS_Y_IX]
#         res[VEL_X_NORMALIZE_INDICE] -= res[PELVIS_VEL_X_IX]
#         res[VEL_Y_NORMALIZE_INDICE] -= res[PELVIS_VEL_Y_IX]
#
#         # # accelerations
#         # res[-VEL_X_NORMALIZE_INDICE.size:] -= res[-VEL_X_NORMALIZE_INDICE.size]
#
#         res[PSOAS_IX] -= 1.0
#
#         return res.tolist()
#
#     def get_aug_dim(self):
#         return BODY_PARTS_IX.size + VEL_X_NORMALIZE_INDICE.size
#
#     def reset(self):
#         self.last_observation = None
#         self.last_velocity = None
#         self.obstacle_pos.clear()


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
        res[X_NORMALIZE_INDICES] -= res[X_NORMALIZE_INDICES].min()
        res = res.tolist()

#        logger.info("observation:")
#        for i, x in enumerate(res):
#            logger.info("{}: {}".format(i+1, x))
#        logger.info("-"*50)

        # threshold = 0.2
        # if np.abs(res[28] - res[30]) >= threshold:
        #     logger.info("Distance between toes larger than {}".format(threshold))
        return res

    def get_aug_dim(self):
        return self.num_body_parts

    def reset(self):
#        logger.info("ob_processor.reset()")
        self.last_observation = None
        self.obstacle_pos.clear()

    def mirror_ob(self, ob0, action, reward, ob1, done, toe_dist_threshold):
        """
        Exchange left and right body parts (if it is pelvis rotation, negate)

        condition: ONLY IF the distance betwen the 2 toes are larger than toe_dist_threshold
        """

        # sanity check
        to_negate = np.asarray([0, 3])  # pelvis rotations

        r_part_org = np.asarray([6, 7, 8, 12, 13, 14, 30, 31, 34, 35, 37])
        r_part_aug = np.asarray([8, 12]) + ORG_OB_DIM
        r_part = np.append(r_part_org, r_part_aug)

        l_part_org = np.asarray([9, 10, 11, 15, 16, 17, 28, 29, 32, 33, 36])
        l_part_aug = np.asarray([6, 10]) + ORG_OB_DIM
        l_part = np.append(l_part_org, l_part_aug)

        assert r_part.size == l_part.size
        assert np.intersect1d(l_part, r_part).size == 0

        # get indices of experiences that are qualified to mirror
        l_toe_x_pos = ob0[:, TOE_L_IX]
        r_toe_x_pos = ob0[:, TOE_R_IX]
        mask = np.abs(l_toe_x_pos - r_toe_x_pos) >= toe_dist_threshold

        if np.sum(mask) > 0:
            # augment ob0
            aug_ob0 = flip_observation(ob0[mask], to_negate, l_part, r_part)
            ob0 = np.concatenate([ob0, aug_ob0], axis=0)
            # augment ob1
            aug_ob1 = flip_observation(ob1[mask], to_negate, l_part, r_part)
            ob1 = np.concatenate([ob1, aug_ob1], axis=0)
            # augment action
            aug_action = deepcopy(action[mask])
            action = np.concatenate([action, aug_action], axis=0)
            # augment reward
            aug_reward = deepcopy(reward[mask])
            reward = np.concatenate([reward, aug_reward], axis=0)
            # augment done
            aug_done = deepcopy(done[mask])
            done = np.concatenate([done, aug_done], axis=0)

        return ob0, action, reward, ob1, done


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
        res[X_NORMALIZE_INDICES] -= res[X_NORMALIZE_INDICES].min()
        res = res.tolist()

        return res

    def reset(self):
#        logger.info("ob_processor.reset()")
        self.last_observations = [None, None, None, None]
        self.obstacle_pos.clear()
