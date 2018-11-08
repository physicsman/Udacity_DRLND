# Referenced from: https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution

import numpy as np
import heapq as hq
import random
from collections import namedtuple, deque
import pickle

from model import QNetwork, QNetwork_image

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)//4 # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
globals()["Experience"] = Experience
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device ',device)

# Tips & code from: https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
#                   https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution          
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, state_size=None, seed=0, image=False, in_channels=3):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.image = image
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.score = 0.0
        
        #self.Experience = experience
        
        # Q-Network
        if image:
            print('image | device: {}'.format(device))
            self.qnetwork_local = QNetwork_image(action_size, seed, in_channels).to(device)
            self.qnetwork_target = QNetwork_image(action_size, seed, in_channels).to(device)
            #self.optimizer = optim.RMSprop(
            #    self.qnetwork_local.parameters(), lr=0.00025, alpha=0.95, eps=0.01, momentum=0.95)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR, amsgrad=False)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR, amsgrad=False)
        # Make totally sure models have same parameters
        self.hard_update(self.qnetwork_local, self.qnetwork_target)
        # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, seed)
        self.memory = RingBuf(action_size, BUFFER_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.f_step = 0
    
    def save(self, tag, mem=True):
        flocal = 'checkpoint_local_image{}.pth'.format(tag)
        ftarget = 'checkpoint_target_image{}.pth'.format(tag)        
        torch.save(self.qnetwork_local.state_dict(), flocal)
        torch.save(self.qnetwork_target.state_dict(), ftarget)
        if mem:
            self.memory.save(tag)
        
    def load(self, tag, mem=True):
        flocal = 'checkpoint_local_image{}.pth'.format(tag)
        ftarget = 'checkpoint_target_image{}.pth'.format(tag)
        self.qnetwork_local.load_state_dict(torch.load(flocal))
        self.qnetwork_target.load_state_dict(torch.load(ftarget))
        if mem:
            self.memory.load(tag)
        
    def step(self, state, action, reward, next_state, done, eps=0.0, p=1.0):

        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if (len(self.memory) > BATCH_SIZE) and (eps < 1.0):
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Prep data from memory depending on which model is being trained
        if self.image:
            states = torch.from_numpy(np.vstack([[e.state] for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([[e.next_state] for e in experiences if e is not None])).float().to(device)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states | '(1 - dones)' ensures Q value of '0' for terminal state 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model | 'gather' ensures model only trains on Q value for the action seen
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) #- self.score
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def hard_update(self, local_model, target_model):
        """Hard update model parameters.

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

# Ring Buffer to make random data selection faster. (didn't notice much of a speed up)
# Modifed from a blogpost I can't find again :/
class RingBuf:
    def __init__(self, action_size, buffer_size, seed):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.buffer_size = buffer_size
        self.data = [None] * (buffer_size + 1)
        self.start = 0
        self.end = 0
        
    def save(self, tag):
        file = 'memory{}.pkl'.format(tag)
        with open(file, 'wb') as f:
            pickle.dump((self.buffer_size, self.data, self.start, self.end), f, pickle.HIGHEST_PROTOCOL)
            
    def load(self, tag):
        file = 'memory{}.pkl'.format(tag)
        with open(file, 'rb') as f:
            self.buffer_size, self.data, self.start, self.end = pickle.load(f)   
            
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.data[self.end] = e
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
            
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        idx = np.random.randint(0, high=self.__len__(), size=batch_size).tolist()
        experiences = [self.__getitem__(i) for i in idx]
        
        return experiences
            
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

# Experimenting with a random buffer that should represent states with a higher 
# likelihood of having been seen recently. 
class RandBuf:
    def __init__(self, action_size, buffer_size, seed):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.buffer_size = buffer_size
        self.data = []
        
    def save(self, tag):
        file = 'memory{}.pkl'.format(tag)
        with open(file, 'wb') as f:
            pickle.dump((self.buffer_size, self.data), f, pickle.HIGHEST_PROTOCOL)
            
    def load(self, tag):
        file = 'memory{}.pkl'.format(tag)
        with open(file, 'rb') as f:
            self.buffer_size, self.data = pickle.load(f)   
            
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        if len(self) == self.buffer_size:
            idx = np.random.randint(self.buffer_size)
            self.data[idx] = e
        else:
            self.data.append(e)
            
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        idx = np.random.randint(0, high=len(self), size=batch_size).tolist()
        experiences = [self.__getitem__(i) for i in idx]
        
        return experiences
            
    def __getitem__(self, idx):
        return self.data[idx % self.buffer_size]
    
    def __len__(self):
        return min(len(self.data), self.buffer_size)
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]