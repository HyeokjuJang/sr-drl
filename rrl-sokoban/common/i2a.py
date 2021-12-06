import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from actor_critic import OnPolicy
from rl import a2c
from config import config

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class RolloutEncoder(nn.Module):
    def __init__(self, in_shape, num_rewards, hidden_size):
        super(RolloutEncoder, self).__init__()
        
        self.in_shape = in_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self.gru = nn.GRU(self.feature_size() + num_rewards, hidden_size)
        
    def forward(self, state, reward):
        num_steps  = state.size(0)
        batch_size = state.size(1)
        
        state = state.view(-1, *self.in_shape)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)
    
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)

class I2A(OnPolicy):
    def __init__(self, in_shape, hidden_size, net, target_net, imagination, emb_size, envs, distillation=True):
        super(I2A, self).__init__()
        
        self.in_shape      = in_shape
        self.net   = net
        self.target_net   = target_net
        self.envs = envs

        self.imagination = imagination
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self.encoder = RolloutEncoder(in_shape, 1, hidden_size)
        self.distillation = distillation
        if self.distillation:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + hidden_size, emb_size),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size(), emb_size),
                nn.ReLU(),
            )
        self.student_weight = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, state, s=None, complete=False):
        batch_size = state.shape[0]
        state_np = state.data.cpu().numpy()
        if s is not None:
            graph_state = s
        else:
            graph_state = self.envs.to_graph(state_np)
        

        imagined_state, imagined_reward = self.imagination(state_np)
        hidden = self.encoder(Variable(imagined_state), Variable(imagined_reward))
        hidden = hidden.view(batch_size, -1)
        
        state = self.features(state)
        state = state.view(state.size(0), -1)
        
        # before knowledge flow version, just concatenate it
        if self.distillation:
            x = torch.cat([state, hidden], 1)
        else:
            x = torch.add(state * self.student_weight, hidden * (1 - self.student_weight))
        x = self.fc(x)
        
        if complete:
            action_softmax, node_softmaxes, value = self.net(graph_state, imag_core_input=x, complete=True)
            return action_softmax, node_softmaxes, value

        action_selected, node_selected, value, tot_prob, a_p, n_p = self.net(graph_state, imag_core_input=x)
        # output shapes [batch_size], [batch_size], [batch_size, 1], [batch_size, 1]
        return action_selected, node_selected, value, tot_prob, a_p, n_p, x
        
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)

    def update(self, r, v, pi, s_, done, state, target_net=None, optimizer=None, entropy_coef=0.01, x=None):
        batch_size = state.shape[0]
        state_np = state.data.cpu().numpy()

        if x is None:
            imagined_state, imagined_reward = self.imagination(state_np)
            hidden = self.encoder(Variable(imagined_state), Variable(imagined_reward))
            hidden = hidden.view(batch_size, -1)
            
            state = self.features(state)
            state = state.view(state.size(0), -1)
            # before knowledge flow version, just concatenate it
            if self.distillation:
                x = torch.cat([state, hidden], 1)
            else:
                x = torch.add(state * self.student_weight, hidden * (1 - self.student_weight))
            
            x = self.fc(x)
        
        done = torch.tensor(done, dtype=torch.float32, device=self.net.device).view(-1, 1)
        r = torch.tensor(r, dtype=torch.float32, device=self.net.device).view(-1, 1)

        if target_net is None:
            target_net = self.net
        v_ = target_net(s_, only_v=True,  imag_core_input=x) * (1. - done)
        
        num_actions = torch.tensor([y[0].shape[0] * 5 for y in s_], dtype=torch.float32, device=self.net.device).reshape(-1, 1) # 5 actions per node
        
        loss, loss_pi, loss_v, loss_h, entropy, log_pi = a2c(r, v, v_, pi, config.gamma, config.alpha_v, self.net.alpha_h, config.q_range, num_actions)
        
        # for logging
        return loss, loss_pi, loss_v, loss_h, entropy, log_pi
    
    def change_env(self, envs):
        self.envs = envs
        self.imagination.envs = envs


    def save(self, file='model.pt'):
        torch.save(self.state_dict(), file)

class ImaginationCore(object):
    def __init__(self, num_rolouts, in_shape, env_model, distil_policy, soko_size, input_frame_shape, envs, num_actions=5):
        self.num_rolouts  = num_rolouts
        self.in_shape      = in_shape
        self.env_model     = env_model
        self.distil_policy = distil_policy
        self.soko_size = soko_size
        self.input_frame_shape = input_frame_shape
        self.envs = envs
        self.num_actions = num_actions
    
    def to_action(self, a, n, s, size):
        node_indices = [x[4] for x in s]

        a = a.cpu().numpy()
        n = n.cpu().numpy()

        nodes = [indices[n[i]] for i, indices in enumerate(node_indices)]

        actions = [ ((nodes[i][1], nodes[i][0]), a[i]) for i in range(len(a)) ] # requires ( (x, y), action )
        return actions

    def __call__(self, state):
        # state, frame info
        batch_size = state.shape[0]

        rollout_states  = []
        rollout_rewards = []

        rollout_batch_size = batch_size
        
        # step action
        for step in range(self.num_rolouts):
            # pick action
            graph_state = self.envs.to_graph(state)
            with torch.no_grad():
                a, n, v, pi, _, _ = self.distil_policy(graph_state)
            actions = self.to_action(a, n, graph_state, size=self.soko_size)
            
            onehot_actions = torch.zeros(batch_size, self.num_actions, self.in_shape[1], self.in_shape[2])
            
            # action embedding
            for i, action in enumerate(actions):
                be_pos, be_a = action 
                onehot_actions[i, be_a, be_pos[1], be_pos[0]] = 1
            
            inputs = torch.cat((torch.from_numpy(state), onehot_actions), dim=1)
            
            with torch.no_grad():
                imagined_state, imagined_reward = self.env_model(Variable(inputs))

            # print(imagined_state.shape) # [256, 4, 10, 10]
            # print(imagined_reward.shape) # [256]
            state = (imagined_state>0.5).int().data.cpu().numpy()
            imagined_state_cpu = imagined_state.data.cpu().view(rollout_batch_size, *self.in_shape)

            
            onehot_reward = torch.zeros(rollout_batch_size, 1)
            onehot_reward[range(rollout_batch_size), 0] = imagined_reward.data.cpu()
            # print(onehot_reward.shape) # [256, 1]
            rollout_states.append(imagined_state_cpu.unsqueeze(0))
            rollout_rewards.append(onehot_reward.unsqueeze(0))
            
            
        
        return torch.cat(rollout_states), torch.cat(rollout_rewards)
        
