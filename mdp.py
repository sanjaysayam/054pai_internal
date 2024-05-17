

def policy_iteration(P, R, gamma, theta):
    num_states = P.shape[0]
    num_actions = P.shape[1]

    # Initialize random policy
    policy = np.random.randint(0, num_actions, size=num_states)

    while True:
        # Policy Evaluation
        V = np.zeros(num_states)
        while True:
            delta = 0
            for s in range(num_states):
                v = V[s]
                V[s] = sum([P[s, policy[s], s1] * (R[s, policy[s], s1] + gamma * V[s1]) for s1 in range(num_states)])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            policy[s] = np.argmax([sum([P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(num_states)]) for a in range(num_actions)])
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break

    return policy, V
P = np.array([[[0.5, 0.5, 0], [0, 1, 0], [0.7, 0.3, 0]],
              [[0, 1, 0], [0, 0, 1], [0, 1, 0]],
              [[1, 0, 0], [1, 0, 0], [0, 0, 1]]])
R = np.array([[[1, 1, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 1]]])

gamma = 0.9
theta = 0.0001

policy, V = policy_iteration(P, R, gamma, theta)

print("Optimal Policy:", policy)
print("Optimal Value Function:", V)
