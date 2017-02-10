

class Policy(object):
    def act(self, obs):
        """
        Returns the action of this policy given the observation, and this
        action may be non-deterministic.

        Args:
            obs: observation

        Returns:
            the action of this policy
        """
        raise NotImplementedError
