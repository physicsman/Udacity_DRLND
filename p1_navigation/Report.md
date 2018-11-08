[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Rewards for Navigation"

[image2]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Rewards for Navigation_Pixels"

# Report:

### Learning Algorithm

This navigation problem has the option of two state representations. The first is a state vector representing a few metrics pertaining to the position and type of object relative to the agent and the second state is a forward looking image of the agents environment. For both states a neural network was trained using Double Deep Q-learning (DDQN).  The neural network predicts the Q-value for every action given a state. The action corresponding to the maximum predicted Q-value is then used to select the agents next action. To mitigate against overestimations seen while training DQN agents a second, identical, neural network is initialized at the beginning of training with the same weights. The first neural network (local model) is used to direct the agent during training and the second network (target model) is used to condition the loss of the local model . The problem is framed as Markov decision process and at each time-step during training the state, previous state, action, and reward are saved to memory. The local model trains by periodicity sampling random batches from memory and predicting the best next action for a state and updates its weights based on how close it was to the predictions of the target model. The target model updates its weights by taking a small step toward the updated weights in the local model. In this way the DDQN regularizes itself by ensuring outlier observations do not self reinforce allowing the model to discover an optimal policy. 

### Plot of Rewards

![Rewards for Navigation][image1]

![Rewards for Navigation_Pixels][image2]

### Ideas for Future Work

I began implementing priority experience replay which selects sample observations during training with a bias toward rare events which would be classified by how much disagreement is seen between the local and target model when processing the observation. However it has yet to be fully incorporated into ‘dqn_agent.py’

I played a bit with 3D convolutions and passing multiple frames to extract some short term temporal information and I would like to explore that model further. 

I notice, during training, the agent would get stuck in a loop of forward+backward or left+right. This behavior would bias the memory to contain many duplicates where the model was not receiving a reward thus biasing training. I would like to implement some sort of metric that agnostically down-samples these experiences.  

Note: Unfortunately the given Unity environment for Linux has a memory leak causing it to save the image from each time-step until training crashes 
