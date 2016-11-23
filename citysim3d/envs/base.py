

class Env(object):
    def step(self, action):
        """
        Run one time step of the environment's dynamics

        Args:
            action: numpy array, which should be contained in the action space
                or be clipped by the action space.

        The action is modified in-place with the action that was actually
        taken. For example, this could happen if the action is not contained
        in the action space or if applying it leads to an invalid state.
        """
        raise NotImplementedError

    def get_state(self):
        """
        Returns the state of the environment

        Returns:
            a numpy array.

        Note:
            Resetting the environment with the current state should not affect
            the state of the environment.

        Example:

            >>> state = self.get_state()
            >>> self.reset(state)
            >>> assert np.allclose(state, self.get_state())

        """
        raise NotImplementedError

    def reset(self, state=None):
        """
        Resets the state of the environment

        Args:
            state: numpy array. If it is not specified, the environment is
                reset to an arbitrary state (which may be chosen at random).
        """
        raise NotImplementedError

    def observe(self):
        """
        Returns a tuple of observations even if there is only one observation.

        Returns:
            a tuple of observations, where each observations is a numpy array,
            and the number of observations should match the number of
            sensor_names. The observations should be contained in the
            observation space.
        """
        raise NotImplementedError

    def render(self):
        pass

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        raise NotImplementedError

    @property
    def sensor_names(self):
        """
        Returns a list of strings
        """
        raise NotImplementedError
