import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Agent(object):
    
   
    def __init__(self, shape, gamma=0.99):


        self.Num_States = shape[0]
        self.Num_Actions = shape[1]
        self.discount_factor = gamma
        
        #Normal Gamma Dist Init
        self.NG = np.zeros(shape=shape, dtype=(float,4))
        for state in range(self.Num_States):
            for action in range(self.Num_Actions):
                self.NG[state][action][1] = 3.#3.
                self.NG[state][action][2] = 1.1#1.5  #alpha>1 ensures the normal-gamma dist is well defined
                self.NG[state][action][3] = 0.75#0.75 #high beta to increase the variance of the prior distribution to explore more

                
    
    def argmax(self,a):

        #If action 'a' has only one max it is the argmax, if many exist one will be randomly picked.
    
        indeces = np.where(np.array(a) == np.max(a))[0]
        return np.random.choice(indeces)


    
    def moment_updating(self, state, action, reward, next_state, done):
        NG = self.NG
        mean = NG[state][action][0]
        lamb = NG[state][action][1]
        alpha = NG[state][action][2]
        beta = NG[state][action][3]

        if not done:

            #sample the next best action at next state
            means = NG[next_state, :, 0]
            next_action = self.argmax(means)

            mean_next = NG[next_state][next_action][0]
            lamb_next = NG[next_state][next_action][1]
            alpha_next = NG[next_state][next_action][2]
            beta_next = NG[next_state][next_action][3]

            #the first two moments of samples 
            M1 = reward + self.discount_factor*mean_next
            M2 = reward**2 + 2*self.discount_factor*reward*mean_next + self.discount_factor**2*(((lamb_next + 1)*beta_next)/(lamb_next*(alpha_next-1)) + mean_next**2)

            #updating the distribution
            self.NG[state][action][0] = (lamb*mean+M1)/(lamb+1)
            self.NG[state][action][1] = lamb + 1
            self.NG[state][action][2]= alpha + 0.5
            self.NG[state][action][3]= beta + 0.5*(M2-M1**2) + (lamb*(M1-mean)**2)/(2*(lamb + 1))

            
    
    
    def update(self, state, action, reward, next_state, done):
        #wrapper
        self.moment_updating(state, action, reward, next_state, done)
    
    
    def Q_value_sampling(self, NG, state):
        #Sample one value for each action
        samples = np.zeros(self.Num_Actions)
        
        for i in range(self.Num_Actions):
            mean = NG[state][i][0]
            lamb = NG[state][i][1]
            alpha = NG[state][i][2]
            beta = NG[state][i][3]

            '''
            It is better to sample from the T-student because in this way we don't need to sample first tau and then
            mu, this reduces the variance.
            '''
            

            samples[i] = np.random.standard_t(2*alpha) * np.sqrt(beta / (alpha * lamb)) + mean
        return self.argmax(samples)
    
    
    def select_action(self, state):
        #wrapper
        return self.Q_value_sampling(self.NG, state)


def run(num_episodes, len_episode):
    # Initializing the  environment
    env = gym.make("NChain-v0")
    Num_States = env.observation_space.n
    Num_Actions = env.action_space.n
    Num_Steps = len_episode
    # Score arrays for each phase
    scores1 = np.zeros(num_episodes) 
    scores2 = np.zeros(num_episodes)
    # Actual discounted rewards for plotting
    rewards = np.zeros(2*Num_Steps)
    discounted_rewards = []
    
    for episode in range(num_episodes):
        agent=Agent(shape=(Num_States, Num_Actions))
        
        #learning phases
        for p in range(2):
            # Reset the environment
            obv = env.reset()
            # the initial state
            state_0 =obv
            
            score=0
            
            done = False
            
            i=0
            for i in range(Num_Steps):
                # Select the action
                action = agent.select_action(state_0)
                
                # Observe the result
                obv, reward, done, _ = env.step(action)
                score+=reward
                rewards[p*1000+i] = reward*agent.discount_factor**i
                
                # Update belief
                state = obv
                agent.update(state_0, action, reward, state, done)
                
                # Setting up for the next iteration
                state_0 = state
                if done:
                    print("Episode {}:Phase {} finished after {} time steps with score={}".format(episode+1 ,p+1, i+1, score))
                    break
                    
            # Store each phase's scores 
            if p == 0:
                scores1[episode]=score
            else:
                scores2[episode]=score
                
        # Store each episode's discounted reward
        discounted_rewards.append(rewards)
        
    # Calculate the mean of the discounted rewards     
    real_discounted_reward = np.zeros(2000)
    
    for reward in discounted_rewards:
        real_discounted_reward += reward
        
    real_discounted_reward //= num_episodes
    
    # Calculate the cumulative sum of 10 runs rewards to plot
    rew = np.cumsum(real_discounted_reward)
    print(100 * "-",'\n')
    
#     print("Avg Score Phase1:%f| std Phase 1:%f|Avg Score Phase2:%f| std Phase 2:%f|" % (np.mean(scores1), np.std(scores1), np.mean(scores2), np.std(scores2)))
    print(20*' ',"Phase 1",20*' ',"Phase 2")
    print("Avg.:",15*" ",np.mean(scores1),20*" ",np.mean(scores2))
    print("Dev.:",15*" ","{:.1f}".format(np.std(scores1)),22*" ","{:.1f}".format(np.std(scores2)))
    
    plt.style.use('bmh')
    _, ax = plt.subplots()
    ax.plot(rew, label='Bays QS+Mom')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accumulated Discounted Reward')
    ax.legend(loc='lower right')

    plt.show()
    
if __name__ == "__main__":
    print("Testing Begins")
    run(10, 1000)
