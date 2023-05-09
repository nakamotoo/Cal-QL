import d4rl
import gym
import numpy as np
import collections

ENV_CONFIG = {
    "antmaze": {
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "adroit-binary": {
        "reward_pos": 0.0,
        "reward_neg": -1.0,
    }
}

class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._mc_returns = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done, mc_returns=-1):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)
        self._mc_returns[self._next_idx] = mc_returns

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
            mc_returns=self._mc_returns[indices, ...],
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...],
            mc_returns=self._mc_returns[:self._size, ...]
        )

# based on https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/__init__.py
def get_d4rl_dataset_with_mc_calculation(env, reward_scale, reward_bias, clip_action, gamma):
    if "antmaze" in env:
        is_sparse_reward=True
    else:
        raise NotImplementedError
    dataset = qlearning_dataset_and_calc_mc(gym.make(env).unwrapped, reward_scale, reward_bias, clip_action, gamma, is_sparse_reward=is_sparse_reward)

    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
        mc_returns=dataset['mc_returns']
    )


def get_hand_dataset_with_mc_calculation(env_name, gamma, add_expert_demos=True, add_bc_demos=True, reward_scale=1.0, reward_bias=0.0, pos_ind=-1, clip_action=None):
    assert env_name in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0", "pen-binary", "door-binary", "relocate-binary"]

    expert_demo_paths = {
        "pen-binary-v0": "demonstrations/offpolicy_hand_data/pen2_sparse.npy",
        "door-binary-v0":"demonstrations/offpolicy_hand_data/door2_sparse.npy",
        "relocate-binary-v0":"demonstrations/offpolicy_hand_data/relocate2_sparse.npy",
    }

    bc_demo_paths = {
        "pen-binary-v0": "demonstrations/offpolicy_hand_data/pen_bc_sparse4.npy",
        "door-binary-v0":"demonstrations/offpolicy_hand_data/door_bc_sparse4.npy",
        "relocate-binary-v0":"demonstrations/offpolicy_hand_data/relocate_bc_sparse4.npy",
    }
    def truncate_traj(env_name, dataset, i, reward_scale, reward_bias, gamma, start_index=None, end_index=None):
        """
        This function truncates the i'th trajectory in dataset from start_index to end_index.
        Since in Adroit-binary datasets, we have trajectories like [-1, -1, -1, -1, 0, 0, 0, -1, -1] which transit from neg -> pos -> neg,
        we truncate the trajcotry from the beginning to the last positive reward, i.e., [-1, -1, -1, -1, 0, 0, 0]
        """
        reward_pos = ENV_CONFIG["adroit-binary"]["reward_pos"]
        
        observations = np.array(dataset[i]["observations"])[start_index:end_index]
        next_observations = np.array(dataset[i]["next_observations"])[start_index:end_index]
        rewards = dataset[i]["rewards"][start_index:end_index]
        dones = (rewards == reward_pos)
        rewards = rewards * reward_scale + reward_bias
        actions = np.array(dataset[i]["actions"])[start_index:end_index]
        mc_returns = calc_return_to_go(env_name, rewards, dones, gamma, reward_scale, reward_bias, is_sparse_reward=True)

        return dict(
                    observations=observations,
                    next_observations=next_observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    mc_returns=mc_returns,
                )

    dataset_list=[]
    dataset_bc_list=[]
    if add_expert_demos:
        print("loading expert demos from:", expert_demo_paths[env_name])
        dataset = np.load(expert_demo_paths[env_name], allow_pickle=True)

        for i in range(len(dataset)):
            N = len(dataset[i]["observations"])
            for j in range(len(dataset[i]["observations"])):
                dataset[i]["observations"][j] = dataset[i]["observations"][j]['state_observation']
                dataset[i]["next_observations"][j] = dataset[i]["next_observations"][j]['state_observation']
            if np.array(dataset[i]["rewards"]).shape != np.array(dataset[i]["terminals"]).shape:
                dataset[i]["rewards"] = dataset[i]["rewards"][:N]

            if clip_action:
                dataset[i]["actions"] = np.clip(dataset[i]["actions"], -clip_action, clip_action)

            assert np.array(dataset[i]["rewards"]).shape == np.array(dataset[i]["terminals"]).shape
            dataset[i].pop('terminals', None)

            if not (0 in dataset[i]["rewards"]):
                continue

            trunc_ind = np.where(dataset[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_traj(env_name, dataset, i, reward_scale, reward_bias, gamma, start_index=None, end_index=trunc_ind)
            dataset_list.append(d_pos)


    if add_bc_demos:
        print("loading BC demos from:", bc_demo_paths[env_name])
        dataset_bc = np.load(bc_demo_paths[env_name], allow_pickle=True)
        for i in range(len(dataset_bc)):
            dataset_bc[i]["rewards"] = dataset_bc[i]["rewards"].squeeze()
            dataset_bc[i]["dones"] = dataset_bc[i]["terminals"].squeeze()
            dataset_bc[i].pop('terminals', None)
            if clip_action:
                dataset_bc[i]["actions"] = np.clip(dataset_bc[i]["actions"], -clip_action, clip_action)

            if not (0 in dataset_bc[i]["rewards"]):
                continue
            trunc_ind = np.where(dataset_bc[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_traj(env_name, dataset_bc, i, reward_scale, reward_bias, gamma, start_index=None, end_index=trunc_ind)
            dataset_bc_list.append(d_pos)

    dataset = np.concatenate([dataset_list, dataset_bc_list])
    
    print("num offline trajs:", len(dataset))
    concatenated = {}
    for key in dataset[0].keys():
        if key in ['agent_infos', 'env_infos']:
            continue
        concatenated[key] = np.concatenate([batch[key] for batch in dataset], axis=0).astype(np.float32)
    return concatenated

def qlearning_dataset_and_calc_mc(env, reward_scale, reward_bias, clip_action, gamma, dataset=None, terminate_on_end=False, is_sparse_reward=True, **kwargs):

    dataset = env.get_dataset(**kwargs)
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True
    
    # first process by traj
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep or i == N-1:
            # Skip this transition and don't apply terminals on the last step of an episode
            pass
        else:
            for k in dataset:
                if k in ['actions', 'next_observations', 'observations', 'rewards', 'terminals', 'timeouts']:
                    data_[k].append(dataset[k][i])
            if 'next_observations' not in dataset.keys():
                data_['next_observations'].append(dataset['observations'][i+1])
            episode_step += 1

        if (done_bool or final_timestep) and episode_step > 0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            episode_data["rewards"] = episode_data["rewards"] * reward_scale + reward_bias
            episode_data["mc_returns"] = calc_return_to_go(env.spec.name, episode_data["rewards"], episode_data["terminals"], gamma, reward_scale, reward_bias, is_sparse_reward)
            episode_data['actions'] = np.clip(episode_data['actions'], -clip_action, clip_action)
            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)
    return concatenate_batches(episodes_dict_list)


def calc_return_to_go(env_name, rewards, terminals, gamma, reward_scale, reward_bias, is_sparse_reward):
    """
    A config dict for getting the default high/low rewrd values for each envs
    This is used in calc_return_to_go func in sampler.py and replay_buffer.py
    """
    if len(rewards) == 0:
        return np.array([])
    
    if "antmaze" in env_name:
        reward_neg = ENV_CONFIG["antmaze"]["reward_neg"] * reward_scale + reward_bias
    elif env_name in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0", "pen-binary", "door-binary", "relocate-binary"]:
        reward_neg = ENV_CONFIG["adroit-binary"]["reward_neg"] * reward_scale + reward_bias
    else:
        assert not is_sparse_reward, "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg): 
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory 
        return_to_go = [float(reward_neg / (1-gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i-1] = rewards[-i-1] + gamma * prev_return * (1 - terminals[-i-1])
            prev_return = return_to_go[-i-1]

    return np.array(return_to_go, dtype=np.float32)

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)

def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated

