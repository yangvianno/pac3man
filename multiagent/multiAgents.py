# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Initialize the evaluation score with the successor state's score
        score = successorGameState.getScore()

        # Calculate the distance to the closest food
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        if foodDistances: score += 10.0 / min(foodDistances)  # Prioritize closer food

        # Calculate the distance to the ghosts and adjust the score based on their positions and scared times
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
          ghostPos = ghostState.getPosition()
          ghostDistance = manhattanDistance(newPos, ghostPos)
            
          if scaredTime > 0: score += 200.0 / (ghostDistance + 1) # If the ghost is scared, prioritize moving towards it to potentially eat it
          else: # If the ghost is not scared, avoid getting too close to it
            if ghostDistance <= 1: score -= 1000.0  # Large penalty for being too close to a ghost
            else: score -= 10.0 / ghostDistance  # Smaller penalty based on distance
            
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def minimax(agentIndex, depth, gameState):
      # Check if we have reached the terminal state or the maximum depth
      if gameState.isWin() or gameState.isLose() or depth == self.depth: return self.evaluationFunction(gameState)

      # Get the number of agents
      numAgents = gameState.getNumAgents()

      # Determine if the current agent is Pacman (maximizer) or a ghost (minimizer)
      isPacman = (agentIndex == 0)

      # Get legal actions for the current agent
      legalActions = gameState.getLegalActions(agentIndex)

      # If there are no legal actions, return the evaluation function for the current state
      if not legalActions:
          return self.evaluationFunction(gameState)

      # Initialize best score
      if isPacman:
          bestScore = float('-inf')
          for action in legalActions:
              successorState = gameState.generateSuccessor(agentIndex, action)
              score = minimax(1, depth, successorState)
              bestScore = max(bestScore, score)
          return bestScore
      else:
          bestScore = float('inf')
          nextAgentIndex = (agentIndex + 1) % numAgents
          nextDepth = depth + 1 if nextAgentIndex == 0 else depth
          for action in legalActions:
              successorState = gameState.generateSuccessor(agentIndex, action)
              score = minimax(nextAgentIndex, nextDepth, successorState)
              bestScore = min(bestScore, score)
          return bestScore

    # Get legal actions for Pacman
    legalActions = gameState.getLegalActions(0)
    bestAction = None
    bestScore = float('-inf')

    # Evaluate each action
    for action in legalActions:
        successorState = gameState.generateSuccessor(0, action)
        score = minimax(1, 0, successorState)
        if score > bestScore:
            bestScore = score
            bestAction = action

    return bestAction










class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
          with alpha-beta pruning.
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Terminal test or max depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # === Pacman’s turn: maximize ===
            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    succ = state.generateSuccessor(agentIndex, action)
                    value = max(value,
                                alphaBeta(succ, depth, 1, alpha, beta))
                    # β-cutoff
                    if value > beta: return value
                    alpha = max(alpha, value)
                return value

            # === Ghost’s turn: minimize ===
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            value = float('inf')
            for action in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, action)
                value = min(value,
                            alphaBeta(succ, nextDepth, nextAgent, alpha, beta))
                # α-cutoff
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        # Top-level: pick the best action for Pacman (agentIndex 0)
        alpha, beta = float('-inf'), float('inf')
        bestScore, bestAction = float('-inf'), None

        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = alphaBeta(succ, 0, 1, alpha, beta)
            if score > bestScore:
                bestScore, bestAction = score, action
            # early β-cutoff at the root
            if bestScore > beta:
                break
            alpha = max(alpha, bestScore)

        return bestAction










class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction