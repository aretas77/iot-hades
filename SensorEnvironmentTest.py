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
        ts = self.env.reset()
        np.testing.assert_array_equal(np.array((0, 1.), np.float32),
                                      ts.observation)

    def test_validate_specs(self):
        utils.validate_py_environment(self.env, episodes=1)

    def test_check_delta_within_bounds(self):
        self.assertEqual(
            (True, -1),
            self.env._check_delta_within_bounds(np.array([2, 0]), 2))
        self.assertEqual(
            (True, 1),
            self.env._check_delta_within_bounds(np.array([1, 0]), 2))
        self.assertEqual(
            (False, 0),
            self.env._check_delta_within_bounds(np.array([1.3, 0]), 2))
        self.assertEqual(
            (False, 0),
            self.env._check_delta_within_bounds(np.array([1.5, 0]), 2))
        return

    def test_calculate_read_time(self):
        self.assertEqual(5, self.env.calculate_read_time(0, np.array([0, 5])))
        self.assertEqual(4, self.env.calculate_read_time(1, np.array([0, 5])))
        self.assertEqual(6, self.env.calculate_read_time(2, np.array([0, 5])))

    def test_add_temperature(self):
        self.env.add_temperature(5)
        self.assertEqual(5, self.env._states[0])


if __name__ == '__main__':
    test_utils.main()
