import sys
sys.path.append("./common")

import numpy as np
import gym, gym_sokoban, torch

import wandb, argparse, itertools, os

from vec_env.subproc_vec_env import SubprocVecEnv
from net import Net
from tqdm import tqdm

# from torchviz import make_dot

from config import config

# ----------------------------------------------------------------------------------------
def to_action(a, n, s, size):
	node_indices = [x[4] for x in s]

	a = a.cpu().numpy()
	n = n.cpu().numpy()

	nodes = [indices[n[i]] for i, indices in enumerate(node_indices)]

	actions = [ ((nodes[i][1], nodes[i][0]), a[i]) for i in range(len(a)) ] # requires ( (x, y), action )
	return actions

def decay_time(step, start, min, factor, rate):
	exp = step / rate * factor
	value = (start - min) / (1 + exp) + min

	return value

def decay_exp(step, start, min, factor, rate):
	exp = step / rate
	value = (start - min) * (factor ** exp) + min

	return value

def init_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def get_args():
	cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

	# optimal cpu=2, device=cuda (rate 3.5)
	parser = argparse.ArgumentParser()
	parser.add_argument('-id', type=str, default="no_id", help="id of the log")

	parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'] + cuda_devices, default='cuda', help="Which device to use")
	parser.add_argument('-cpus', type=str, default='4', help="How many CPUs to use")
	parser.add_argument('-batch', type=int, default=256, help="Size of a batch / How many CPUs to use")
	parser.add_argument('-seed', type=int, default=None, help="Random seed") # seed in multiprocessing is not implemented
	parser.add_argument('-load_model', type=str, default=None, help="Load model from this file")
	parser.add_argument('-max_epochs', type=int, default=None, help="Terminate after this many epochs")
	parser.add_argument('-mp_iterations', type=int, default=10, help="Number of message passes")
	parser.add_argument('-epoch', type=int, default=1000, help="Epoch length")
	parser.add_argument('-eval_problems', type=int, default=1000, help="Epoch length")
	parser.add_argument('-save_dir', type=str, default="~/dataset/sr-drl/models/", help="save log directory")
	
	parser.add_argument('-subset', type=int, default=None, help="Use a subset of train set")
	parser.add_argument('--pos_feats', action='store_const', const=True, help="Enable positional features")
	parser.add_argument('--custom', type=str, default=None, help="Custom size (e.g. 10x10x4; else Boxoban)")

	parser.add_argument('-trace', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('-trace_i2a', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('-eval', action='store_const', const=True, help="Evaluate the agent")
	parser.add_argument('-env_pretrain', action='store_const', const=True, help="Pretrain env model")
	parser.add_argument('-pretrained_env_test', action='store_const', const=True, help="test Pretrained env model")
	

	parser.add_argument('--num_rollouts', type=int, default=5,
                    help='num of rollouts for i2a algorithm')
	parser.add_argument('--num_steps', type=int, default=32,
                    help='num of steps for update')
	parser.add_argument('--debug', action='store_true',
                    help='connect to wandb')
	parser.add_argument('--distillation', action='store_true',
                    help='work with distillation, or knowledge flow')
	parser.add_argument('--distil_policy_equal_net', action='store_true',
                    help='distill policy and net are the same model')
	parser.add_argument('--d_alone', type=int, default=2,
                    help='distillation learn alone times in interval')
	parser.add_argument('--d_interval', type=int, default=10,
                    help='distillation learn alone interval')

	cmd_args = parser.parse_args()

	return cmd_args

# ----------------------------------------------------------------------------------------
def evaluate(net, split='valid', subset=None):
	test_env = SubprocVecEnv([lambda: gym.make('Sokograph-v0', split=split, subset=subset) for i in range(config.eval_batch)], in_series=(config.eval_batch // config.cpus), context='fork')
	tqdm_val = tqdm(desc='Validating', total=config.eval_problems, unit=' steps')

	with torch.no_grad():
		net.eval()

		r_tot = 0.
		problems_solved = 0
		problems_finished = 0
		steps = 0

		s = test_env.reset()

		while problems_finished < config.eval_problems:
			steps += 1

			a, n, v, pi, _, _ = net(s)
			actions = to_action(a, n, s, size=config.soko_size)

			s, r, d, i = test_env.step(actions)

			# print(r)
			r_tot += np.sum(r)
			problems_solved   += sum('all_boxes_on_target' in x and x['all_boxes_on_target'] == True for x in i)
			problems_finished += np.sum(d)

			tqdm_val.update()

		r_avg = r_tot / (steps * config.eval_batch) # average reward per step
		problems_solved_ps  = problems_solved / (steps * config.eval_batch)
		problems_solved_avg = problems_solved / problems_finished

		net.train()

	tqdm_val.close()
	test_env.close()

	return r_avg, problems_solved_ps, problems_solved_avg, problems_finished

def evaluate_i2a(net, split='valid', subset=None, student_only=False):
	tqdm_val = tqdm(desc='Validating', total=config.eval_problems, unit=' steps')
	test_env = SubprocVecEnv([lambda: gym.make('Sokograph-v0', split=split, subset=subset) for i in range(config.eval_batch)], in_series=(config.eval_batch // config.cpus), context='fork')
	tmp_env = net.envs
	net.change_env(test_env)
	if student_only:
		tmp_student_weight = net.student_weight
		net.student_weight = torch.tensor(1.0)
	with torch.no_grad():
		net.eval()

		r_tot = 0.
		problems_solved = 0
		problems_finished = 0
		steps = 0

		s = net.envs.reset()
		
		while problems_finished < config.eval_problems:
			state_as_frame = Variable(torch.tensor(net.envs.raw_state(), dtype=torch.float))
			steps += 1
			
			a, n, v, pi, a_p, n_p, _ = net(state_as_frame, s=s) # action, node, value, total_prob
			actions = to_action(a, n, s, size=config.soko_size)

			s, r, d, i = net.envs.step(actions)

			# print(r)
			r_tot += np.sum(r)
			problems_solved   += sum('all_boxes_on_target' in x and x['all_boxes_on_target'] == True for x in i)
			problems_finished += np.sum(d)

			tqdm_val.update()

		r_avg = r_tot / (steps * config.eval_batch) # average reward per step
		problems_solved_ps  = problems_solved / (steps * config.eval_batch)
		problems_solved_avg = problems_solved / problems_finished

		net.train()
	net.change_env(tmp_env)
	if student_only:
		net.student_weight = tmp_student_weight
	tqdm_val.close()

	return r_avg, problems_solved_ps, problems_solved_avg, problems_finished

# ----------------------------------------------------------------------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_plotly(net, env, s, title=None, distil=True):
	s_img = env.render(mode='rgb_array')
	
	if distil:
		action_softmax, node_softmaxes, value = net([s], complete=True)
	else:
		state_as_frame = Variable(torch.tensor([net.envs.raw_state()], dtype=torch.float))
		action_softmax, node_softmaxes, value = net(state_as_frame, s=[s], complete=True)

	action_probs = action_softmax.probs[0].cpu()
	# node_probs = node_softmaxes[0].reshape(*config.soko_size, 5).flip(0).cpu() # flip is because Heatmap flips the display :-(
	value = value[0].item()

	node_indices = s[4]
	node_probs = np.zeros((*config.soko_size, 5))
	node_probs[tuple(node_indices.T)] = node_softmaxes[0].cpu()
	node_probs = np.flip(node_probs, 0)

	fig = make_subplots(rows=2, cols=4, subplot_titles=["State", "MOVE_TO", "PUSH_UP", "State value", "Action probs", "PUSH_LEFT", "PUSH_DOWN", "PUSH_RIGHT"])

	fig.add_trace(go.Image(z=s_img), 1, 1)
	fig.add_trace(go.Bar(x=["MOVE_TO", "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT"], y=action_probs), 2, 1)
	fig.add_trace(go.Bar(x=["value"], y=[value], text=[f'{value:.2f}'], textposition='auto', width=[0.2]), 1, 4)
	fig.update_yaxes(range=config.q_range, row=1, col=4)

	fig.add_trace(go.Heatmap(z=node_probs[:, :, 0], zmin=0., zmax=1., colorscale='Greys', showscale=False), 1, 2)	# MOVE_TO
	fig.add_trace(go.Heatmap(z=node_probs[:, :, 1], zmin=0., zmax=1., colorscale='Greys', showscale=False), 1, 3)	# PUSH_UP
	fig.add_trace(go.Heatmap(z=node_probs[:, :, 2], zmin=0., zmax=1., colorscale='Greys', showscale=False), 2, 3) # PUSH_DOWN
	fig.add_trace(go.Heatmap(z=node_probs[:, :, 3], zmin=0., zmax=1., colorscale='Greys', showscale=False), 2, 2) # PUSH_LEFT
	fig.add_trace(go.Heatmap(z=node_probs[:, :, 4], zmin=0., zmax=1., colorscale='Greys', showscale=True),  2, 4) # PUSH_RIGHT

	fig.update_layout(showlegend=False, title=title, title_x=0.5)

	return fig, value, action_probs

def debug_net(net, distil=True, student_only=False):
	test_env = gym.make('Sokograph-v0', split='valid')
	# test_env = SubprocVecEnv([lambda: gym.make('Sokograph-v0', split='valid') for i in range(config.eval_batch)], in_series=(config.eval_batch // config.cpus), context='fork')
	if distil:
		s = test_env.reset()
	else:
		tmp_env = net.envs
		net.change_env(test_env)
		s = net.envs.reset()
	
	if student_only:
		tmp_student_weight = net.student_weight
		net.student_weight = torch.tensor(1.0)

	
	with torch.no_grad():
		net.eval()
		fig, value, action_probs = get_plotly(net, test_env, s, distil=distil)
		net.train()
	
	if not distil:
		net.change_env(tmp_env)
	if student_only:
		net.student_weight = tmp_student_weight

	if distil:
		wandb.log({'net_debug': fig, 'value': value, 
			'aprob_goto': action_probs[0], 'aprob_up': action_probs[1], 'aprob_down': action_probs[2], 
			'aprob_left': action_probs[3], 'aprob_right': action_probs[4]}, commit=False)
	else:
		wandb.log({'net_debug_i2a': fig, 'value_i2a': value, 
			'aprob_goto_i2a': action_probs[0], 'aprob_up_i2a': action_probs[1], 'aprob_down_i2a': action_probs[2], 
			'aprob_left_i2a': action_probs[3], 'aprob_right_i2a': action_probs[4]}, commit=False)

def trace_net(net, net_name, steps=100):
	import imageio, io
	from pdfrw import PdfReader, PdfWriter

	test_env = gym.make('Sokograph-v0', split='valid')
	s = test_env.reset()

	with torch.no_grad():
		net.eval()
		imgs = []

		tqdm_trace = tqdm(desc='Creating trace', unit=' steps', total=steps)
		for step in range(steps):
			fig, _, _ = get_plotly(net, test_env, s, title=f"{net_name} | step {test_env.step_idx} ({test_env.num_env_steps})")
			imgs.append(fig.to_image(format='pdf'))

			# make a regular step
			a, n, v, pi = net([s])
			actions = to_action(a, n, [s], size=config.soko_size)
			s, r, d, i = test_env.step(actions[0])

			tqdm_trace.update()

		writer = PdfWriter()
		for img in imgs:
			pdf_img = PdfReader(io.BytesIO(img)).pages
			writer.addpages(pdf_img)

		writer.write('trace.pdf')

		net.train()

def trace_i2a_net(net, net_name, steps=100):
	import imageio, io
	from pdfrw import PdfReader, PdfWriter

	test_env = gym.make('Sokograph-v0', split='valid')
	
	tmp_env = net.envs
	net.change_env(test_env)
	s = net.envs.reset()

	with torch.no_grad():
		net.eval()
		imgs = []

		tqdm_trace = tqdm(desc='Creating trace', unit=' steps', total=steps)
		for step in range(steps):
			fig, _, _ = get_plotly(net, test_env, s, title='{}-{}-steps'.format(net_name, str(step)), distil=False)
			imgs.append(fig.to_image(format='pdf'))

			state_as_frame = Variable(torch.tensor([net.envs.raw_state()], dtype=torch.float))
			a, n, v, pi, a_p, n_p, _ = net(state_as_frame, s=[s])
			
			actions = to_action(a, n, [s], size=config.soko_size)

			s, r, d, i = net.envs.step(actions[0])

			tqdm_trace.update()

		writer = PdfWriter()
		for img in imgs:
			pdf_img = PdfReader(io.BytesIO(img)).pages
			writer.addpages(pdf_img)

		writer.write('trace.pdf')

		net.change_env(tmp_env)
		net.train()

def env_pretrain(net, tot_steps, subset=None, ):
	from common.environment_model import EnvModelSokoban as EnvModel

	test_env = SubprocVecEnv([lambda: gym.make('Sokograph-v0',  subset=subset) for i in range(config.eval_batch)], in_series=(config.eval_batch // config.cpus), context='fork')
	tqdm_val = tqdm(desc='Training', total=tot_steps, unit=' steps')
	
	net.eval()

	steps = 0
	reward_coef = 0.1
	num_actions = 5

	s = test_env.reset()
	input_state = test_env.raw_state()
	input_state_shape = input_state.shape
	
	num_pixels = int(input_state_shape[1])
	env_model = EnvModel(input_state_shape[1:4], num_pixels, 1)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(env_model.parameters())

	losses = []

	while steps < tot_steps:
		steps += 1
		input_state = torch.tensor(test_env.raw_state()) # arr_walls, arr_goals, arr_boxes, arr_player
		
		a, n, v, pi = net(s) # action, node, value, total_prob
		
		actions = to_action(a, n, s, size=config.soko_size)
		
		onehot_actions = torch.zeros(input_state_shape[0], num_actions, input_state_shape[2], input_state_shape[3])
	
		# action embedding
		for i, action in enumerate(actions):
			be_pos, be_a = action 
			onehot_actions[i, be_a, be_pos[1], be_pos[0]] = 1
		
		inputs = torch.autograd.Variable(torch.cat((input_state, onehot_actions), dim=1))
		
		imagined_state, imagined_reward = env_model(inputs)
		
		s, r, d, i = test_env.step(actions)

		target_state = torch.autograd.Variable(torch.tensor(test_env.raw_state(), dtype=torch.float))
		target_reward = torch.autograd.Variable(torch.tensor(r, dtype=torch.float))
			
		optimizer.zero_grad()
		image_loss  = criterion(imagined_state, target_state)
		reward_loss = criterion(imagined_reward, target_reward)
		loss = image_loss + reward_coef * reward_loss
		
		loss.backward()
		optimizer.step()


		losses.append(loss.item())
        
		if steps % 10000 == 0:
			print('epoch %s. loss: %s' % (steps, losses[-1]))

		tqdm_val.update()

	torch.save(env_model.state_dict(), "env_model_sokoban")

	net.train()

	tqdm_val.close()
	test_env.close()

	return env_model

def pretrained_env_test(net, tot_steps, subset=None, ):
	from common.environment_model import EnvModelSokoban as EnvModel
	import matplotlib.pyplot as plt
	import time

	test_env = SubprocVecEnv([lambda: gym.make('Sokograph-v0',  subset=subset) for i in range(1)], context='fork')
	
	net.eval()

	steps = 0
	num_actions = 5

	s = test_env.reset()
	input_state = test_env.raw_state()
	input_state_shape = input_state.shape
	
	num_pixels = int(input_state_shape[1])
	env_model = EnvModel(input_state_shape[1:4], num_pixels, 1)
	env_model.load_state_dict(torch.load("env_model_sokoban"))
	
	def target_to_pix(i_s):
		pixels = []
		# arr_walls, arr_goals, arr_boxes, arr_player
		
		for i in range(i_s.shape[1]):
			for j in range(i_s.shape[2]):
				if (i_s[:,i,j] == [1, 0, 0, 0]).all():
					pixels.append((0, 0, 0))
				elif (i_s[:,i,j] == [0, 0, 0, 0]).all():
					pixels.append((243, 248, 238))
				elif (i_s[:,i,j] == [0, 1, 0, 0]).all():
					pixels.append((254, 126, 125))
				elif (i_s[:,i,j] == [0, 0, 1, 0]).all():
					pixels.append((254, 95, 56))
				elif (i_s[:,i,j] == [0, 1, 1, 0]).all():
					pixels.append((142, 121, 56))
				elif (i_s[:,i,j] == [0, 0, 0, 1]).all():
					pixels.append((160, 212, 56))
				elif (i_s[:,i,j] == [0, 1, 0, 1]).all():
					pixels.append((219, 212, 56))
				else:
					pixels.append((255, 255, 255))
				
		return np.array(pixels).reshape(10,10,3)

	while steps < tot_steps:
		steps += 1
		input_state = torch.tensor(test_env.raw_state()) # arr_walls, arr_goals, arr_boxes, arr_player
		
		a, n, v, pi = net(s) # action, node, value, total_prob
		
		actions = to_action(a, n, s, size=config.soko_size)
		
		onehot_actions = torch.zeros(input_state_shape[0], num_actions, input_state_shape[2], input_state_shape[3])
	
		# action embedding
		for i, action in enumerate(actions):
			be_pos, be_a = action 
			onehot_actions[i, be_a, be_pos[1], be_pos[0]] = 1
		
		inputs = torch.cat((input_state, onehot_actions), dim=1)
		imagined_state, imagined_reward = env_model(inputs)
		imagined_state = (imagined_state>0.5).int().reshape(4,10,10)
		imagined_image = target_to_pix(imagined_state.data.cpu().numpy())
		
		s, r, d, i = test_env.step(actions)
		state_image = target_to_pix(test_env.raw_state().reshape(4,10,10))
		
		
		plt.figure(figsize=(10,3))
		plt.subplot(121)
		plt.title("Imagined")
		plt.imshow(imagined_image)
		plt.subplot(122)
		plt.title("Actual")
		plt.imshow(state_image)
		plt.show()
		time.sleep(0.3)

	test_env.close()

	return env_model

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
	from common.actor_critic import OnPolicy, ActorCritic, RolloutStorage
	from common.environment_model import EnvModelSokoban as EnvModel
	from common.i2a import ImaginationCore, I2A
	import torch.optim as optim
	import torch.nn.functional as F

	args = get_args()
	config.init(args)

	print(f"Config: {config}")

	gym.envs.registration.register(
		id='Sokograph-v0',
		entry_point='graph_sokoban:GraphSokoban',
		kwargs={'difficulty': 'unfiltered', 'max_steps': config.soko_max_steps, 'pos_feats': config.pos_feats, 
			'boxoban': config.boxoban, 'soko_size': config.soko_size, 'soko_boxes': config.soko_boxes}
	)

	if config.seed:
		init_seed(config.seed)

	torch.set_num_threads(config.cpus)	
	
	net = Net(inside_i2a=True, distillation=args.distillation)
	target_net = Net(inside_i2a=True, distillation=args.distillation)
	if args.distil_policy_equal_net:
		distil_policy = net
	else:
		distil_policy = Net()
	if args.distillation:
		distil_target_policy = Net()

	if config.load_model:
		distil_policy.load(os.path.join(config.load_model, "files", "model_distil.pt"))
		
		print(f"Model loaded: {config.load_model}")

	if args.trace:
		trace_net(distil_policy, config.load_model)
		exit(0)

	if args.eval:
		r_avg, s_ps_avg, s_avg, s_tot = evaluate(distil_policy)
		print(f"Avg. reward: {r_avg}, Avg. solved per step: {s_ps_avg}, Avg. solved: {s_avg}, Tot. finished: {s_tot}")
		exit(0)

	if args.env_pretrain:
		print("start env model pre-training")
		env_pretrain(distil_policy, 1e6)
		exit(0)
	
	if args.pretrained_env_test:
		print("test pre-trained env model")
		pretrained_env_test(distil_policy, 1e4)
		exit(0)

	envs = SubprocVecEnv([lambda: gym.make('Sokograph-v0', subset=config.subset) for i in range(config.batch)], in_series=(config.batch // config.cpus), context='fork')
	# env = ParallelEnv('Sokograph-v0', n_envs=N_ENVS, cpus=N_CPUS)

	job_name = f'{config.soko_size[0]}x{config.soko_size[1]}-{config.soko_boxes} '\
				f'mp-{config.mp_iterations} nn-{config.emb_size} b-{config.batch} '\
				f'id-{args.id}' \
				# f'd_times-{config.distil_learn_alone} d_interval-{config.distil_learn_alone_interval} '\
	
	debug = args.debug
	if not debug:
		wandb.init(project="sokoban_i2a_sr-drl", name=job_name, config=config, dir=config.save_dir)
		wandb.save("*.pt")

		wandb.watch(net, log='all')
	# print(net)

	USE_CUDA = torch.cuda.is_available()
	Variable = lambda *args, **kwargs: torch.autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else torch.autograd.Variable(*args, **kwargs)

	s = envs.reset()
	input_state = envs.raw_state()
	input_state_shape = input_state.shape
	state_shape = input_state_shape[1:4]

	num_pixels = int(input_state_shape[1])
	env_model = EnvModel(state_shape, num_pixels, 1)

	env_model.load_state_dict(torch.load("env_model_sokoban"))

	imagination = ImaginationCore(args.num_rollouts, state_shape, env_model, distil_policy, config.soko_size, input_state, envs)
	actor_critic = I2A(state_shape, 256, net, target_net, imagination, config.emb_size, envs, args.distillation)
	
	# rmsprop hyperparams:
	lr = 7e-4
	eps = 1e-5
	alpha = 0.99
	# rmsprop
	# optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
	# adam:
	optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
	if args.distillation:
		distil_optimizer = distil_policy.opt

	if USE_CUDA:
		env_model = env_model.cuda()
		net = net.cuda()
		target_net = target_net.cuda()
		actor_critic = actor_critic.cuda()
		distil_policy = distil_policy.cuda()
		if args.distillation:
			distil_target_policy = distil_target_policy.cuda()
	gamma = 0.99
	entropy_coef = 0.01
	value_loss_coef = 0.5
	max_grad_norm = 0.5
	num_steps = args.num_steps
	num_frames = int(10e6)

	rollout = RolloutStorage(num_steps, config.batch, state_shape)
	rollout.cuda()

	tot_env_steps = 0
	tot_el_env_steps = 0
	
	if config.load_model:
		actor_critic.load_state_dict(torch.load(os.path.join(config.load_model, "files","model_i2a.pt")))

	if args.trace_i2a:
		trace_i2a_net(actor_critic, "I2A + SR-DRL")
		exit(0)

	tqdm_main = tqdm(desc='Training', unit=' steps')
	s = envs.reset()
	state_as_frame = Variable(torch.tensor(envs.raw_state(), dtype=torch.float))
	torch.autograd.set_detect_anomaly(True)
	for step in itertools.count(start=1):
		
		a, n, v, pi, a_p, n_p, imag_core_input = actor_critic(state_as_frame, s=s) # action, node, value, total_prob
		a_d, n_d, v_d, pi_d, a_p_d, n_p_d = distil_policy(s)
		
		# draw graph
		# make_dot(distil_policy(s), params=dict(distil_policy.named_parameters())).render("graph", format="png")

		if (step % config.distil_learn_alone_interval) < config.distil_learn_alone and args.distillation:
			actions = to_action(a_d, n_d, s, size=config.soko_size)
		else:
			actions = to_action(a, n, s, size=config.soko_size)
		
		# print(actions)
		s, r, d, i = envs.step(actions)
		
		s_true = [x['s_true'] for x in i]
		d_true = [x['d_true'] for x in i]
		state_as_frame = Variable(torch.tensor(envs.raw_state(), dtype=torch.float))
		# update network
		if args.distillation:
			if (step % config.distil_learn_alone_interval) < config.distil_learn_alone:
				loss, loss_pi, loss_v, loss_h, entropy = distil_policy.update(r, v_d, pi_d, s_true, d_true, distil_target_policy)
				distil_target_policy.copy_weights(distil_policy, rho=config.target_rho)
				distil_optimizer.zero_grad()
				loss.backward()
				distil_grad_norm = torch.nn.utils.clip_grad_norm_(distil_policy.parameters(), max_grad_norm)

				distil_optimizer.step()
			else:
				loss, loss_pi, loss_v, loss_h, entropy, logit = actor_critic.update(r, v, pi, s_true, d_true, state_as_frame, target_net, optimizer)
				target_net.copy_weights(net, rho=config.target_rho)
		
				# distillation
				distil_loss_action = F.kl_div(F.log_softmax(a_p_d + 1e-5, dim=1), F.softmax(a_p + 1e-9, dim=1).detach())
				distil_loss_node = F.kl_div(F.log_softmax(n_p_d + 1e-5, dim=1), F.softmax(n_p + 1e-9, dim=1).detach()) 
				distil_loss_pi = F.kl_div(torch.log(pi_d), pi.detach())
				distil_loss_value = F.mse_loss(v.detach(), v_d)
				# distil_loss_entropy = (torch.log(a_p_d + 1e-5).mean() + torch.log(n_p_d + 1e-5).mean()) * entropy_coef
				distil_loss_entropy = (torch.log(pi_d) + 1e-5).mean() * entropy_coef
				optimizer.zero_grad()
				loss = loss - entropy
				loss.backward()
				norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), config.opt_max_norm)
				optimizer.step()
				
				distil_optimizer.zero_grad()
				distil_loss = distil_loss_action + distil_loss_node + distil_loss_value + distil_loss_pi + distil_loss_entropy
				distil_loss.backward()
				distil_grad_norm = torch.nn.utils.clip_grad_norm_(distil_policy.parameters(), max_grad_norm)

				distil_optimizer.step()
		else:
			loss, loss_pi, loss_v, loss_h, entropy, logit = actor_critic.update(r, v, pi, s_true, d_true, state_as_frame, target_net, optimizer, x=imag_core_input)
			target_net.copy_weights(net, rho=config.target_rho)
			# knowledge flow dependency loss
			loss_dep = -torch.log(actor_critic.student_weight)/2
			optimizer.zero_grad()
			loss = loss - entropy + loss_dep
			loss.backward()
			norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), config.opt_max_norm)
			optimizer.step()

			if not args.distil_policy_equal_net:
				# copy actor_critic weights to distil_policy
				distil_policy.copy_weights(actor_critic.net, rho=config.target_rho)

		# save step stats
		tot_env_steps += config.batch
		tot_el_env_steps += np.sum([x['elementary_steps'] for x in i])

		tqdm_main.update()

		if step % config.sched_lr_rate == 0:
			lr = decay_exp(step, config.opt_lr, config.sched_lr_min, config.sched_lr_factor, config.sched_lr_rate)
			net.set_lr(lr)

		if step % config.sched_alpha_h_rate == 0:
			alpha_h = decay_time(step, config.alpha_h, config.sched_alpha_h_min, config.sched_alpha_h_factor, config.sched_alpha_h_rate)
			net.set_alpha_h(alpha_h)

		if step % config.log_rate == 0:
			log_step = step // config.log_rate

			r_avg, s_ps_avg, s_avg, _ = evaluate(distil_policy)
			r_avg_trn, s_ps_avg_trn, s_avg_trn, _ = evaluate(distil_policy, split='train', subset=config.subset)
			debug_net(distil_policy)
			
			r_avg_i2a, s_ps_avg_i2a, s_avg_i2a, _ = evaluate_i2a(actor_critic)
			debug_net(actor_critic, distil=False)
			if args.distillation:
				log = {
					'env_steps': tot_env_steps,
					'el_env_steps': tot_el_env_steps,

					'rate': tqdm_main.format_dict['rate'],
					'loss': loss,
					'loss_pi': loss_pi,
					'loss_v': loss_v,
					'loss_h': loss_h,
					'distil_node_loss': distil_loss_action,
					'distil_action_loss': distil_loss_node,
					'distil_value_loss': distil_loss_value,
					'distil_pi_loss': distil_loss_pi,
					'distil_loss': distil_loss,
					'entropy estimate': entropy,
					'gradient norm': norm,
					'distil_gradient_norm': distil_grad_norm,

					'distil_value': v_d.mean(),

					'lr': net.lr,
					'alpha_h': net.alpha_h,

					'reward_avg': r_avg,
					'solved_per_step': s_ps_avg,
					'solved_avg': s_avg,

					'reward_avg_i2a': r_avg_i2a,
					'solved_per_step_i2a': s_ps_avg_i2a,
					'solved_avg_i2a': s_avg_i2a,

					'reward_avg_train': r_avg_trn,
					'solved_per_step_train': s_ps_avg_trn,
					'solved_avg_train': s_avg_trn
				}
			else:
				log = {
					'env_steps': tot_env_steps,
					'el_env_steps': tot_el_env_steps,

					'rate': tqdm_main.format_dict['rate'],
					'loss': loss,
					'loss_pi': loss_pi,
					'loss_v': loss_v,
					'loss_h': loss_h,
					'entropy estimate': entropy,
					'gradient norm': norm,
					'student_weight': actor_critic.student_weight,
					'teacher_weight': 1.0 - actor_critic.student_weight,

					'lr': net.lr,
					'alpha_h': net.alpha_h,

					'reward_avg': r_avg,
					'solved_per_step': s_ps_avg,
					'solved_avg': s_avg,

					'reward_avg_i2a': r_avg_i2a,
					'solved_per_step_i2a': s_ps_avg_i2a,
					'solved_avg_i2a': s_avg_i2a,

					'reward_avg_train': r_avg_trn,
					'solved_per_step_train': s_ps_avg_trn,
					'solved_avg_train': s_avg_trn
				}

			print(log)
			wandb.log(log)

			# save model to wandb
			actor_critic.save(os.path.join(wandb.run.dir, "model_i2a.pt"))
			distil_policy.save(os.path.join(wandb.run.dir, "model_distil.pt"))

		# finish if max_epochs exceeded
		if config.max_epochs and (step // config.log_rate >= config.max_epochs):
			break

	envs.close()
	tqdm_main.close()