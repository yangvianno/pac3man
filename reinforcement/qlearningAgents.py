# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        ReinforcementAgent.__init__(self, **args)  # Call parent class initializer
        self.qValues = {}  # Dictionary to store Q-values for (state, action) pairs

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.qValues:
          self.qValues[(state, action)] = random.uniform(-0.01, 0.01)  # Initialize to a small random value
        return self.qValues.get((state, action), 0.0)


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0

        maxQValue = float("-inf")
        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                maxQValue = qValue

        return maxQValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        bestAction = None
        maxQValue = float("-inf")
        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action

        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # # Pick Action
        # legalActions = self.getLegalActions(state)
        # action = None
        # "*** YOUR CODE HERE ***"
        # if not legalActions:
        #   return None

        # # Epsilon-greedy strategy for exploration vs exploitation
        # if random.random() < self.epsilon: 
        #     action = random.choice(legalActions)  # Explore: pick a random action
        # else:
        #   # Exploit: choose action with the highest Q-value
        #   best_value = float("-inf")
        #   best_action = None

        #   for action in legalActions:
        #       q_value = self.getQValue(state, action)
        #       if q_value > best_value:
        #           best_value = q_value
        #           best_action = action

        #   action = best_action if best_action is not None else random.choice(legalActions)

        # # Debugging print statements
        # print(f"GetAction: state={state}, action={action}, epsilon={self.epsilon}")

        # # Update lastState
        # self.lastState = state
        # return action
    
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Epsilon-greedy strategy for exploration vs exploitation
        if random.random() < self.epsilon:
            action = random.choice(legalActions)  # Explore: pick a random action
        else:
            # Exploit: choose action with the highest Q-value
            bestAction = self.computeActionFromQValues(state)
            action = bestAction

        # Debugging print statements
        print(f"GetAction: state={state}, action={action}, epsilon={self.epsilon}")

        # Update lastState
        self.lastState = state
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # q_value = self.getQValue(state, action)
        # next_max_q_value = self.computeValueFromQValues(nextState)
        # sample = reward + self.discount * next_max_q_value
        # self.qValues[(state, action)] = (1 - self.alpha) * q_value + self.alpha * sample

        # Get the current Q-value
        currentQValue = self.getQValue(state, action)
        
        # Compute the maximum Q-value for the next state
        nextMaxQValue = self.computeValueFromQValues(nextState)
        
        # Compute the updated Q-value
        updatedQValue = (1 - self.alpha) * currentQValue + self.alpha * (reward + self.discount * nextMaxQValue)
        
        # Update the Q-value in the dictionary
        self.qValues[(state, action)] = updatedQValue

        # Debugging print statements
        print(f"Update: state={state}, action={action}, reward={reward}, nextState={nextState}")
        print(f"Q-value: current={currentQValue}, nextMax={nextMaxQValue}, updated={updatedQValue}")
        print(f"Q-values: {self.qValues}")

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
