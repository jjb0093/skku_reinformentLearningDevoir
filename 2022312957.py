import numpy as np

def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    while(1):
        while(1):
            # Policy Evaluation 단계
            variation = 0
            for state in range(env.nS):
                value = 0
                for action in range(env.nA):
                    for prob, nextState, reward in env.MDP[state][action]:
                        value += policy[state][action] * prob * (reward + gamma * V[nextState])

                variation = max(variation, abs(value - V[state]))
                V[state] = value

            if(variation < theta): break

        # Policy Improvement 단계
        policyStable = True
        for state in range(env.nS):
            oldAction = np.argmax(policy[state])

            qValues = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, nextState, reward in env.MDP[state][action]:
                    qValues[action] += prob * (reward + gamma * V[nextState])

            newAction = np.argmax(qValues)
            newPolicy = np.zeros(env.nA)
            newPolicy[newAction] = 1

            policy[state] = newPolicy

            if(oldAction != newAction): policyStable = False

        if(policyStable): break

    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    while(1):
        variation = 0
        for state in range(env.nS):
            qValues = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, nextState, reward in env.MDP[state][action]:
                    qValues[action] += prob * (reward + gamma * V[nextState])

            bestValue = np.max(qValues)
            variation = max(variation, abs(bestValue - V[state]))
            V[state] = bestValue

        if(variation < theta): break

    newPolicy = np.zeros([env.nS, env.nA])
    for state in range(env.nS):
        qValues = np.zeros(env.nA)
        for action in range(env.nA):
            for prob, nextState, reward in env.MDP[state][action]:
                qValues[action] += prob * (reward + gamma * V[nextState])

        bestAction = np.argmax(qValues)
        newPolicy[state][bestAction] = 1.0

    policy = newPolicy

    return policy, V