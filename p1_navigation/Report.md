[//]: # (Image References)

[image1]: https://github.com/physicsman/Udacity_DRLND/blob/master/p1_navigation/rewards_plot_navigation.png "Rewards for Navigation"

# Report:

### Learning Algorithm

This navigation problem has the option of two state representations. The first is a state vector representing a few metrics pertaining to the position and type of object relative to the agent and the second state is a forward looking image of the agents environment. For both states a neural network was trained using Double Deep Q-learning (DDQN).  The neural network predicts the Q-value for every action given a state. The action corresponding to the maximum predicted Q-value is then used to select the agents next action. To mitigate against overestimations seen while training DQN agents a second, identical, neural network is initialized at the beginning of training with the same weights. The first neural network (local model) is used to direct the agent during training and the second network (target model) is used to condition the loss of the local model . The problem is framed as Markov decision process and at each time-step during training the state, previous state, action, and reward are saved to memory. The local model trains by periodicity sampling random batches from memory and predicting the best next action for a state and updates its weights based on how close it was to the predictions of the target model. The target model updates its weights by taking a small step toward the updated weights in the local model. In this way the DDQN regularizes itself by ensuring outlier observations do not self reinforce allowing the model to discover an optimal policy. 

### Plot of Rewards

![Rewards for Navigation][image1]

### Hyperparameters and Architecture 

For the 1D case the hyperparameters were chosen to match those given in an example problem from there I adjusted each parameter individually to look for simple improvements however, even though the problem being solved is different the set of example parameters performed best. 

The architecture for the 1D case neural network is a simple feedforward network with three linear layers, and RELU activations to add nonlinearity representation.  

For the 2D case a much lower ‘eps_start’ seemed to help the model train. I’m not sure how the CNN initialization coupled to exploration but letting the CNN pick actions early on was better than random choice. 

The architecture for the 2D case neural network is a modest multilayer CNN each layer consists of two 2D convolutions. The first 2D convolution lengthens and the second squeezes the stack of features. In between the activations are regularized with batchnormilization and nonlinearity is added via SELU activations. A global average pooling layer compresses the final stack of features into a 1D vector which is then fed into a simple stack of linear layers to output the final action value pairs.

There seems to be a lot that can be accomplished by tunning here however at this point exploration feels more like dart throwing than learning. 

### Ideas for Future Work

I began implementing priority experience replay which selects sample observations during training with a bias toward rare events which would be classified by how much disagreement is seen between the local and target model when processing the observation. However it has yet to be fully incorporated into ‘dqn_agent.py’

I played a bit with 3D convolutions and passing multiple frames to extract some short term temporal information and I would like to explore that model further. 

I notice, during training, the agent would get stuck in a loop of forward+backward or left+right. This behavior would bias the memory to contain many duplicates where the model was not receiving a reward thus biasing training. I would like to implement some sort of metric that agnostically down-samples these experiences.  

Note: Unfortunately the given Unity environment for Linux has a memory leak causing it to save the image from each time-step until training crashes 
