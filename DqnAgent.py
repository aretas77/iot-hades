import os
import logging

try:
    import tensorflow as tf
    try:
        from SensorEnvironment import SensorEnv
        from tf_agents.trajectories import trajectory
        from tf_agents.environments import tf_py_environment
        from tf_agents.agents.dqn import dqn_agent
        from tf_agents.networks import q_network
        from tf_agents.replay_buffers import tf_uniform_replay_buffer as replay
        from tf_agents.policies import policy_saver
        from tf_agents.utils import common
    except ImportError:
        print("failed to import libraries")
except ImportError:
    print("failed to import tensorflow or numpy")


class DqnAgent:

    # Hyperparameters for the network
    learning_rate = 1
    log_interval = 1
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000
    initial_step = True

    def __init__(self):
        # Dictionaries that keep parts of DqnAgent for different devices
        self.devices = []
        self.env = {}
        self.train_env = {}
        self.train_checkpointer = {}
        self.policy_saver = {}
        self.global_step = {}
        self.agent = {}
        self.q_net = {}
        self.eval_policy = {}
        self.collect_policy = {}
        self.checkpoint_dirs = {}
        self.policy_dirs = {}
        self.replay_buffer = {}
        self.initial_training = {}

        self.checkpoint_dir = "checkpoints"
        self.policy_dir = "policies"

        logging.info("initialized DqnAgent")
        return

    def device_exists(self, mac):
        if mac in self.devices:
            return True
        return False

    def add_device(self, mac):
        self.devices.append(mac)
        self.initial_training[mac] = True

        # construct directory names
        self.checkpoint_dirs[mac] = os.path.join(self.checkpoint_dir, mac)
        self.policy_dirs[mac] = os.path.join(self.policy_dir, mac)

        # initialize environment for the device
        self._init_env(mac)

        # create the QNetwork for the device
        self.q_net[mac] = q_network.QNetwork(
                self.train_env[mac].observation_spec(),
                self.train_env[mac].action_spec())

        # create a global step (required for checkpoints)
        self.global_step[mac] = tf.compat.v1.train.get_or_create_global_step()

        self._init_agent(mac, self.learning_rate, self.q_net[mac],
                         self.global_step[mac], self.train_env[mac])
        self._init_policy(mac, self.agent[mac])
        self._init_replay_buffer(mac, self.agent[mac], self.train_env[mac])
        self._init_checkpointer(mac, self.agent[mac], self.replay_buffer[mac],
                                self.global_step[mac])
        self._init_policy_saver(mac, self.agent[mac])

        logging.info("added a device with MAC = " + mac)
        return

    def _init_env(self, mac):
        """Will initialize a custom made Python Environment. This is a step
        zero for subsequent initializations.
        """
        self.env[mac] = SensorEnv(mac)
        self.train_env[mac] = tf_py_environment.TFPyEnvironment(self.env[mac])

        return

    def _init_agent(self, mac, learning_rate, q_net, train_step_counter,
                    train_env):
        optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate)

        self.agent[mac] = dqn_agent.DqnAgent(
                train_env.time_step_spec(),
                train_env.action_spec(),
                q_network=q_net,
                optimizer=optimizer,
                td_errors_loss_fn=common.element_wise_squared_loss,
                train_step_counter=train_step_counter)
        self.agent[mac].initialize()

        return

    def _init_policy(self, mac, agent):
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

            We use the default ones, atlthough, it would be better to make
            custom ones.
        """
        self.eval_policy[mac] = agent.policy
        self.collect_policy[mac] = agent.collect_policy

        return

    def _init_replay_buffer(self, mac, agent, train_env):
        """Replay buffer keeps track of data collected from the environment.
        We will be using TFUniformReplayBuffer.
        """
        self.replay_buffer[mac] = replay.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=self.replay_buffer_max_length)

        return

    def _init_checkpointer(self, mac, agent, replay_buffer, global_step):
        """Create a checkpointer for a given device with identification of
        a given MAC address. We will use this to save/load the training state
        of the device.
        """
        self.train_checkpointer[mac] = common.Checkpointer(
            ckpt_dir=self.checkpoint_dirs[mac],
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step)

        return

    def _init_policy_saver(self, mac, agent):
        self.policy_saver[mac] = policy_saver.PolicySaver(agent.policy)

        return

    def convert_to_tflite(self, mac):
        """convert_to_tflite loads up the policy of the MAC address and tries
        to convert it to the TensorFlow Lite model using concrete function for
        policy 'action'. However, in current TensorFlow Lite implementation
        some ops used here are not yet supported: BroadcastArgs and BroadcastTo
        """
        export_dir = os.path.join(self.policy_dir, mac)

        model = tf.saved_model.load(export_dir=export_dir)
        concrete_func = model.signatures['action']

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_func])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        model_dir = os.path.join('models', mac)
        open(model_dir, 'wb').write(tflite_model)

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
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step,
                                          next_time_step)
        buffer.add_batch(traj)

    def train(self, mac):
        """Here we run the training for the given device. The data is extracted
        from the environment using collect_step and given to the network for
        training. The trained networks checkpoint and policies are then saved.
        """

        # don't allow training for devices that don't exist here.
        if mac not in self.devices:
            return

        if self.initial_training[mac]:
            self.agent[mac].train_step_counter.assign(0)
            self.initial_training[mac] = False
        else:
            self.train_checkpointer[mac].initialize_or_restore()
            self.global_step[mac] = tf.compat.v1.train.get_global_step()

        # collect data
        self.collect_step(self.train_env[mac], self.agent[mac].collect_policy,
                          self.replay_buffer[mac])

        dataset = self.replay_buffer[mac].as_dataset(
            num_parallel_calls=2,
            sample_batch_size=2,
            num_steps=2).prefetch(2)
        iterator = iter(dataset)

        experience, unused_info = next(iterator)
        _ = self.agent[mac].train(experience).loss

        self.train_checkpointer[mac].save(self.global_step[mac])
        self.policy_saver[mac].save(self.policy_dirs[mac])
        self.convert_to_tflite(mac)
