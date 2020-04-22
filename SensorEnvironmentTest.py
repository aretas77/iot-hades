import numpy as np
from SensorEnvironment import SensorEnv
from tf_agents.environments import utils
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.utils import test_utils


class SensorEnvironmentTest(test_utils.TestCase):

    def setUp(self):
        super(SensorEnvironmentTest, self).setUp()
        self.discount = np.asarray(1., dtype=np.float32)
        self.env = SensorEnv()

        # Get reward values
        self.REWARD_WITHIN_BOUNDS = self.env.REWARD_WITHIN_BOUNDS
        self.REWARD_DO_NOTHING = self.env.REWARD_DO_NOTHING
        self.REWARD_OUT_OF_BOUNDS = self.env.REWARD_OUT_OF_BOUNDS

        ts = self.env.reset()
        np.testing.assert_array_equal(np.array((0), np.float32),
                                      ts.observation)

    # def test_validate_specs(self):
    #    utils.validate_py_environment(self.env, episodes=1)

    def test_check_delta_within_bounds(self):
        self.assertEqual(
            (True, self.REWARD_OUT_OF_BOUNDS),
            self.env._check_delta_within_bounds(np.array([2]), 2))
        self.assertEqual(
            (True, self.REWARD_WITHIN_BOUNDS),
            self.env._check_delta_within_bounds(np.array([1]), 2))
        self.assertEqual(
            (False, self.REWARD_DO_NOTHING),
            self.env._check_delta_within_bounds(np.array([1.3]), 2))
        self.assertEqual(
            (False, self.REWARD_DO_NOTHING),
            self.env._check_delta_within_bounds(np.array([1.5]), 2))
        return

    def test_calculate_read_time(self):
        self.assertEqual(5, self.env.calculate_read_time(0, 5))
        self.assertEqual(4, self.env.calculate_read_time(1, 5))
        self.assertEqual(6, self.env.calculate_read_time(2, 5))

    def test_check_calculated(self):
        # check if when a negative reward is given and the action was 1, it
        # means that we decremented the send_interval so reward positively.
        self.assertEqual(
            (self.REWARD_WITHIN_BOUNDS, 4),
            self.env._check_calculated(np.array([2]), 2, 4, -2, 5))

        # check if when got a positive reward and we could increase the send
        # interval, but it wasn't done - reward negatively.


if __name__ == '__main__':
    test_utils.main()
