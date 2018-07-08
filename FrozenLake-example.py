import gym
import numpy as np

env = gym.make('FrozenLake-v0')

#Initialize the Q-learning table with zeros

Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
alpha = .8
gamma = .95
num_episodes = 2000

# Create lists to contain total rewards per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s0 = env.reset()
    rAll = 0
    terminated = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j+=1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s0,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # Get new state and reward from environment
        s1,r,terminated,_ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s0,a] = Q[s0,a] + alpha*(r + gamma*np.max(Q[s1,:]) - Q[s0,a])
        rAll += r
        s0 = s1
        if terminated == True:
            break
    rList.append(rAll)

print("Score over time: " +  str(sum(rList)/num_episodes))

print("Final Q-Table Values")
print(Q)
