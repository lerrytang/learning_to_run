import numpy as np
import h5py
import chainer
import sys
import os


# converts serialized Keras model parameters into Chainer's serialization format.
# Only supports fully-connected layer.


def convert_keras_chainer(keras_h5_path, dst_chainer_npz_path, chainer_chain, layer_name_map):
    k_dataset = h5py.File(keras_h5_path, mode="r")

    tmp_npz_path = dst_chainer_npz_path + ".tmp"
    chainer.serializers.save_npz(tmp_npz_path, chainer_chain)
    c_weight_dict = dict(np.load(tmp_npz_path))

    overwritten_keys = set()

    for k_name, c_name in layer_name_map.items():
        try:
            weight_entry = k_dataset["{}/{}/kernel:0".format(k_name, k_name)]
            bias_entry = k_dataset["{}/{}/bias:0".format(k_name, k_name)]
        except KeyError:
            sys.stderr.write("Layer {} does not exist in keras model.\n".format(k_name))
            continue

        try:
            c_initial_weight = c_weight_dict["{}/W".format(c_name)]
            c_initial_bias = c_weight_dict["{}/b".format(c_name)]
        except KeyError:
            sys.stderr.write("Layer {} does not exist in chainer model.\n".format(c_name))
            continue

        c_initial_weight[:] = weight_entry.value.T  # keras: (in_channels, out_channels) / chainer: (out, in)
        c_initial_bias[:] = bias_entry.value
        overwritten_keys.add("{}/W".format(c_name))
        overwritten_keys.add("{}/b".format(c_name))

    untouched_keys = set(c_weight_dict.keys()) - overwritten_keys
    if len(untouched_keys) > 0:
        sys.stderr.write(
            "These items in chainer model are not overwritten by keras parameter: {}\n".format(untouched_keys))

    np.savez(dst_chainer_npz_path, **c_weight_dict)
    os.remove(tmp_npz_path)


def convert_sample():
    # This is example of calling convert_keras_chainer with fixed parameters.
    # User should call convert_keras_chainer from own trainer to give properly initialized chainer.Chain.
    from nipsenv import NIPS
    import trpo.models
    env = NIPS(visualize=False)
    obs_space = (env.observation_space.shape[0] + 14,)  # 14 is BodySpeedAugmentor().get_aug_dim()
    policy_hiddens = [128, 128, 64, 64]
    policy = trpo.models.GaussianMLPPolicy(
        observation_space=obs_space,
        action_space=env.action_space,
        env_spec=env.spec,
        action_nonlinearity=chainer.functions.tanh,
        hidden_sizes=policy_hiddens,
        hidden_nonlinearity=chainer.functions.relu,
    )

    layer_name_map = {"action": "l_act"}  # keras name: chainer name
    for i in range(1, 5):
        # actor_fc1: fc1
        layer_name_map["actor_fc{}".format(i)] = "fc{}".format(i)
    in_path = "../trials/candidate_01/actor.h5"
    out_path = "../trials/candidate_01/actor_chainer.npz"
    sys.stderr.write("Converting keras model {} into chainer model {}\n".format(in_path, out_path))
    convert_keras_chainer(in_path, out_path, policy, layer_name_map)
    sys.stderr.write("done\n")


if __name__ == "__main__":
    convert_sample()
