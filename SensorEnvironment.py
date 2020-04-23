import numpy as np
import tensorflow as tf
import os
import json
from hades_utils import num
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class SensorEnv(py_environment.PyEnvironment):
    """A state-settable-gettable environment for statistics send interval
    optimization.

    We keep the state of the environment in a array so we could restore or set
    it any time.

    The states is a vector of two elements. The first indicating the delta
    temperature which is a delta of previously got temperature reading and
    currently received temperature reading while the second is a current
    temperature send interval.
    The action is a vector of one element which indicates what action to take
    with the current temperature send interval.
    """
    REWARD_WITHIN_BOUNDS = np.asarray(1., dtype=np.float32)
    REWARD_DO_NOTHING = np.asarray(0., dtype=np.float32)
    REWARD_OUT_OF_BOUNDS = np.asarray(-2., dtype=np.float32)

    REWARD_CORRECT_ACTION = np.asarray(1., dtype=np.float32)
    REWARD_INCORRECT_ACTION = np.asarray(-2., dtype=np.float32)

    REWARD_WITHIN_BOUNDS.setflags(write=False)
    REWARD_OUT_OF_BOUNDS.setflags(write=False)
    REWARD_DO_NOTHING.setflags(write=False)

    REWARD_INCORRECT_ACTION.setflags(write=False)
    REWARD_CORRECT_ACTION.setflags(write=False)

    def __init__(self, mac, discount=0.5, delta=3):
        super(SensorEnv, self).__init__()

        # the environment of a given device with a MAC address.
        self._mac = mac
        self._iteration = 0
        self.state_dir = "states"

        # three actions are allowed for reading time modification:
        #   0. Do nothing
        #   1. Decrease
        #   2. Increase
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=2, name='action')

        # observation is a received temperature from the device and its current
        # send interval. We hard code the minimum temperature to be 0 and
        # maximum as 100 as currently I don't have data on other temperatures.
        # Also, hard code minimum and maximum for send interval in minutes -
        # set minimum as 1 minute while maximum as 86400minutes (24hours).
        # self._observation_spec = array_spec.BoundedArraySpec(
        #    shape=(2,), dtype=np.float32, minimum=[0, 1],
        #    maximum=[100, 3600 * 24], name='observation')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, minimum=[0],
            maximum=[100], name='observation')

        # the value which detones the boundary of temperature delta.
        self._delta = delta

        # state will represent the deltas of:
        #   - Previous received temperature value and the currently received
        #     value.
        #   - Previous received temperature time and currently received time
        #     in short - sensor send interval in minutes.
        # when receiving the first temperature value - we should set the state
        # so that the delta of previous temperature and currently received
        # temperature would be zero and set the default send interval.
        # self._states = [0, 1]

        # states will represent the delta of:
        #   - Previous received temperature value and the currently received
        #     value.
        self._states = [0]
        self._current_send_interval = 0
        self._previous_temperature = -999
        self._current_temperature = -999
        self._line = 1
        self._episode_ended = False

        # we make all rewards equal - in essence, ignore the discount.
        self._discount = np.asarray(discount, dtype=np.float32)

    def action_spec(self):
        """Override the internal specified actions. It will
        define the actions that should be provided to `step()`.
        """
        return self._action_spec

    def observation_spec(self):
        """Override the internal observation specification. It will define
        the structure of observations that are provided by the environment.
        """
        return self._observation_spec

    def _reset(self):
        self._states = [0]
        self._episode_ended = False
        return ts.restart(np.array(self._states, dtype=np.float32))

    def load_env_state(self, mac):
        state_file = os.path.join(self.state_dir, mac)

        with open(state_file, 'r') as json_file:
            data = json.load(json_file)
            self._states[0] = data['stats']['prev_delta']

            previous_temperature = data['stats']['prev_temperature']
            self._previous_temperature = num(previous_temperature)

            current_temperature = data['stats']['curr_temperature']
            self._current_temperature = num(current_temperature)

            current_send_interval = data['stats']['send_interval']
            self._current_send_interval = num(current_send_interval)

        return

    def save_env_state(self, mac, states):
        """We save important environment values in a json dump file for later
        reuse. As when using a Checkpointer it doesn't save environment values.

        Args:
            mac: the MAC address of the device whose state we save.
            states: the state of the environment.
        """
        state_file = os.path.join(self.state_dir, mac)

        data = {}
        data['stats'] = {
            'prev_temperature': self._previous_temperature,
            'prev_delta': self._states[0],
            'curr_temperature': self._current_temperature,
            'send_interval': self._current_send_interval,
        }

        with open(state_file, 'w') as outfile:
            json.dump(data, outfile)

        return

    def _step(self, action):
        self._iteration += 1
        self.load_env_state(self._mac)

        # check if this is a first iteration
        if self._previous_temperature == -999:
            self._previous_temperature = self._current_temperature

        self._states[0] = abs(self._previous_temperature -
                              self._current_temperature)
        self._previous_temperature = self._current_temperature

        # we shouldn't enter this, unless we do it manually
        if self._episode_ended or self._current_time_step.is_last():
            return self.reset()

        # we did a step - have new values, so calculate them
        adjust, reward = self._check_delta_within_bounds(self._states,
                                                         self._delta)
        if adjust:
            # blindly calculate interval
            new_interval = self.calculate_read_time(
                action, self._current_send_interval)

            # we have a new interval - let's confirm whether it is correct.
            reward, self._current_send_interval = self._check_calculated(
                self._states, self._delta, new_interval, reward,
                self._current_send_interval)

        # we save save the environment every other iteration.
        # XXX: figure out the logic behind training the network and why we
        # take two steps, that is - current and next observation.
        if self._iteration % 2 == 0:
            self.save_env_state(self._mac, self._states)

        # terminate immediatly
        return ts.transition(np.array(self._states, dtype=np.float32), reward,
                             self._discount)

    def calculate_read_time(self, action, send_interval):
        """Adjust the read interval according to the action selected by the
        policy.

        TODO: calculate the interval modification time by how much current
        temperature delta is differing from allowed delta.

        Args:
            action: the action to be taken selected by the policy.
            states: the state of temperature and send interval of delta.

        Returns:
            A modified send interval.
        """

        # we need to decrease the read time as the temperature differences
        # are too great.
        if action == 1 and send_interval > 1:
            send_interval -= 1  # decrease by a minute

        # we need to increase the read time as we have some spare room for
        # accuracy disregard.
        if action == 2:
            send_interval += 1  # increase by a minute

        return send_interval

    def _check_delta_within_bounds(self, states, delta):
        """Check if the given state temperature delta is within given delta.

        Args:
            states: states of temperature.

        Returns:
            A tuple of (need_adjust, reward) where was_adjusted means whether
            we need to make changes to send interval or not, and reward is the
            reward for is temperature delta within defined delta.

            The meaning of reward:
            1 = within bounds (good), 0 = just good, -1 = out of bounds (bad)
        """
        currentDelta = abs(states[0])

        # XXX: the 'just in case' values (0.7; 0.5) should be made more logical
        # and derived from something.
        if currentDelta >= delta:
            # temperature delta is out of safe bounds! Reduce send interval.
            # Simple as that!
            return True, self.REWARD_OUT_OF_BOUNDS  # decrease send interval
        elif currentDelta + 0.7 >= delta:
            # mmm, we are in a comfortable position and we can leave it at that
            return False, self.REWARD_DO_NOTHING    # don't adjust
        elif currentDelta < (delta - 0.5):
            # if we come here, it means that we can further increase the send
            # interval. We do this by checking that we are still within delta
            # bounds minus accuracy penalty (just in case by increasing the
            # send interval we make the delta too big).
            return True, self.REWARD_WITHIN_BOUNDS  # increase send interval

        # if nothing is triggered
        return False, self.REWARD_DO_NOTHING

    def _check_calculated(self, states, delta, n_interval, reward, o_interval):
        """Should be called after calculations were made. With this, we do
        additional checking whether the actions taken were logical or not and
        based on the actions taken and depending on the interval and deltas
        we adjust the send interval and reward.

        Args:
            states: state of temperature delta.
            delta: the boundary delta.
            n_interval: the calculated new send interval.
            o_interval: the previous send interval.
            reward: the calculated reward.
        """
        current_delta = states[0]
        new_interval = n_interval
        old_interval = o_interval

        # if we got a negative reward and action was 1 - we reward.
        if new_interval < old_interval and reward < 0:
            return self.REWARD_CORRECT_ACTION, new_interval

        # if we got a positive reward and needed to adjust, it means we can
        # increase the send interval. However, if we haven't adjusted it,
        # punish with negative reward.
        if reward > 0 and new_interval <= old_interval:
            return self.REWARD_INCORRECT_ACTION, new_interval

        # if we increased the send interval but the delta was too big. Reward
        # with a negative reward and decrease the send interval.
        if new_interval > old_interval and delta < current_delta:
            if old_interval > 1:
                return self.REWARD_INCORRECT_ACTION, old_interval - 1
            else:
                return self.REWARD_INCORRECT_ACTION, old_interval

        # if we decremented and the delta was too big - reward!
        if new_interval < old_interval and delta < current_delta:
            return self.REWARD_CORRECT_ACTION, new_interval

        # if we can increase but haven't - negative reward.
        if current_delta < (delta - 0.5) and new_interval == old_interval:
            return self.REWARD_INCORRECT_ACTION, new_interval + 1

        # just in case return
        return reward, new_interval
