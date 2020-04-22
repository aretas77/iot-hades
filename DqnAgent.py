import numpy as np
import tensorflow as tf
from SensorEnvironment import SensorEnv
from tf_agents.trajectories import trajectory
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common


class DqnAgent:

    # Hyperparameters for the network
    fc_layer_params = 100
    learning_rate = 1
    log_interval = 1

    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000

    # parts of our agent
    env = None
    agent = None
    q_net = None
    replay_buffer = None

    initial_training = True
    initial_temperature = True
    initial_step = True
    current_temperature = 0
    previous_temperature = 0
    temperature_delta = 0

    def __init__(self):
        self._init_env()

        # create a QNetwork
        self.q_net = q_network.QNetwork(
                self.train_env.observation_spec(),
                self.train_env.action_spec())

        # setup Deep Q Network agent.
        optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate)
        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
                self.train_env.time_step_spec(),
                self.train_env.action_spec(),
                q_network=self.q_net,
                optimizer=optimizer,
                td_errors_loss_fn=common.element_wise_squared_loss,
                train_step_counter=self.train_step_counter)
        self.agent.initialize()

        self._init_policy(self.agent)
        self._init_replay_buffer(self.agent, self.train_env)
        return

    def _init_env(self):
        """Will initialize a custom made Python Environment. This is a step
        zero for subsequent initializations.
        """
        self.env = SensorEnv()
        self.train_env = tf_py_environment.TFPyEnvironment(self.env)

    def _init_policy(self, agent):
        """
            - In our case, the desired outcome is to keep the temperature
            readings delta within boundaries of a pre-defined delta.
            - The policy returns an action which depicts what to with the
            current send interval (do nothing, increase or decrease).

            The dqn_agent contains two policies:
            - agent.policy is the main policy that is used for evaluation and
            deployment.
            - agent.collect_policy is a second policy that is used for data
            collection.
        """
        self.eval_policy = agent.policy
        self.collect_policy = agent.collect_policy

    def _init_replay_buffer(self, agent, train_env):
        """Replay buffer keeps track of data collected from the environment.
        We will be using TFUniformReplayBuffer.
        """
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=self.replay_buffer_max_length)

    """
        Methods to process the environment.
    """

    def update_temperature(self, temperature):
        if self.initial_temperature:
            self.current_temperature = temperature
            self.previous_temperature = temperature
        else:
            self.temperature_delta = abs(temperature -
                                         self.previous_temperature)

    def collect_step(self, env, policy, buffer):
        """Collects the current time step of the environment and maps the
        current time_step to action in Q-table.
        """
        if self.initial_step:
            self.initial_step = False
            time_step = env.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step,
                                              next_time_step)
            buffer.add_batch(traj)

        time_step = env.current_time_step()
        print(time_step)
        action_step = policy.action(time_step)
        print(action_step)
        print('\n')
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step,
                                          next_time_step)
        buffer.add_batch(traj)

    def train(self):
        if self.initial_training:
            self.agent.train_step_counter.assign(0)
            self.initial_training = False

        for _ in range(26):

            # collect a step
            self.collect_step(self.train_env, self.agent.collect_policy,
                              self.replay_buffer)

            dataset = self.replay_buffer.as_dataset(
                num_parallel_calls=2,
                sample_batch_size=2,
                num_steps=2).prefetch(2)
            iterator = iter(dataset)

            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience).loss
            # print(train_loss)
            # step = self.agent.train_step_counter.numpy()


net = DqnAgent()
net.train()
