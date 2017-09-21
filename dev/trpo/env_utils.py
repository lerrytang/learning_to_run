import multiprocessing as mp
import sys
import subprocess
import numpy as np


def env_worker(env_maker, conn, n_worker_envs):
    envs = []
    for _ in range(n_worker_envs):
        # envs.append(env_maker.make())
        envs.append(env_maker())
    while True:
        command, data = conn.recv()
        try:
            if command == 'reset':
                obs = []
                for env in envs:
                    obs.append(env.reset())
                conn.send(('success', obs))
            elif command == 'seed':
                seeds = data
                for env, seed in zip(envs, seeds):
                    env.seed(seed)
                conn.send(('success', None))
            elif command == 'step':
                actions = data
                results = []
                for env, action in zip(envs, actions):
                    next_ob, rew, done, info = env.step(action)
                    if done:
                        info["last_observation"] = next_ob
                        next_ob = env.reset()
                    results.append((next_ob, rew, done, info))
                conn.send(('success', results))
            elif command == 'close':
                for env in envs:
                    env.close()
                conn.send(('success', None))
                return
            else:
                raise ValueError("Unrecognized command: {}".format(command))
        except Exception as e:
            conn.send(('error', sys.exc_info()))


class EnvPool(object):
    """
    Using a pool of workers to run multiple environments in parallel. This implementation supports multiple environments
    per worker to be as flexible as possible.
    """

    def __init__(self, env_maker, ob_processor_maker, n_envs=mp.cpu_count(), n_parallel=mp.cpu_count()):
        self.env_maker = env_maker
        self.n_envs = n_envs
        # No point in having more parallel workers than environments
        if n_parallel > n_envs:
            n_parallel = n_envs
        self.n_parallel = n_parallel
        self.workers = []
        self.conns = []
        # try to split evenly, but this isn't always possible
        self.n_worker_envs = [len(d) for d in np.array_split(
            np.arange(self.n_envs), self.n_parallel)]
        self.worker_env_offsets = np.concatenate(
            [[0], np.cumsum(self.n_worker_envs)[:-1]])
        self.last_obs = None
        self.ob_processors = [ob_processor_maker() for _ in range(n_envs)]

    def start(self):
        workers = []
        conns = []
        for idx in range(self.n_parallel):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(target=env_worker, args=(
                self.env_maker, worker_conn, self.n_worker_envs[idx]))
            worker.start()
            # pin each worker to a single core
            if sys.platform == 'linux':
                subprocess.check_call(
                    ["taskset", "-p", "-c",
                     str(idx % mp.cpu_count()), str(worker.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            workers.append(worker)
            conns.append(master_conn)

        self.workers = workers
        self.conns = conns

        # set initial seeds
        seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=self.n_envs)
        self.seed([int(x) for x in seeds])

    def __enter__(self):
        self.start()
        return self

    def reset(self):
        for conn in self.conns:
            conn.send(('reset', None))
        obs = []
        for conn in self.conns:
            status, data = conn.recv()
            if status == 'success':
                obs.extend(data)
            else:
                raise data[1].with_traceback(data[2])
        assert len(obs) == self.n_envs

        # reset each ob processor
        processed_obs = [None] * self.n_envs
        for idx in range(self.n_envs):
            self.ob_processors[idx].reset()
            processed_obs[idx] = self.ob_processors[idx].process(obs[idx])
        self.last_obs = processed_obs
        return processed_obs

    def step(self, actions):
        assert len(actions) == self.n_envs
        for idx, conn in enumerate(self.conns):
            offset = self.worker_env_offsets[idx]
            conn.send(
                ('step', actions[offset:offset + self.n_worker_envs[idx]]))

        results = []

        for conn in self.conns:
            status, data = conn.recv()
            if status == 'success':
                results.extend(data)
            else:
                raise data[1].with_traceback(data[2])
        next_obs, rews, dones, infos = list(map(list, zip(*results)))
        new_next_obs = [None] * self.n_envs
        new_infos = [None] * self.n_envs
        for idx in range(self.n_envs):
            new_next_obs[idx] = self.ob_processors[idx].process(next_obs[idx])
            if dones[idx]:
                new_infos[idx] = {'last_observation': self.ob_processors[idx].process(infos[idx]['last_observation'])}
        self.last_obs = new_next_obs
        return new_next_obs, rews, dones, new_infos

    def seed(self, seeds):
        assert len(seeds) == self.n_envs
        for idx, conn in enumerate(self.conns):
            offset = self.worker_env_offsets[idx]
            conn.send(('seed', seeds[offset:offset + self.n_worker_envs[idx]]))
        for conn in self.conns:
            status, data = conn.recv()
            if status != 'success':
                raise data[1].with_traceback(data[2])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        for conn in self.conns:
            conn.send(('close', None))
        for conn in self.conns:
            status, data = conn.recv()
            if status != 'success':
                raise data[1].with_traceback(data[2])
        for worker in self.workers:
            worker.join()
        self.workers = []
        self.conns = []
