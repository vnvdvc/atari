import os,sys
from tensorflow.python.saved_model import tag_constants
from gym import wrappers
from skimage.transform import resize
from skimage.color import rgb2gray
import tensorflow as tf
import numpy as np

def preprocess_frame(frame,shape):
    return resize(rgb2gray(frame),shape)

def preprocess_obs(obs_array):
    if len(obs_array) == 1:
        return obs_array[0]
    else:
        shape = obs_array[0].shape
        if len(shape) < 3:
            return np.stack(obs_array,axis=-1)
        else:
            return np.concatenate(obs_array,axis=-1)

def run_policy(model_path,checkpoint,game,video_dir,record=True):
    #rl config
    shape_of_frame = (84,84)
    num_recent_obs = 4
    action_repeat = 4
    max_steps = 2000
    num_trials = 5

    env = gym.make(game)
    if record:
        env = wrappers.Monitor(env,video_dir)
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        print("Restore model from model_path.")
        tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                model_path
                )
        obs_ph = graph.get_tensor_by_name('obs_ph:0')
        indexed_action_ph = graph.get_tensor_by_name('indexed_action_ph:0')
        y_ph = graph.get_tensor_by_name('y_ph:0')

        preds = graph.get_tensor_by_name('preds:0')
        q = graph.get_tensor_by_name('q:0')

        print("Restore values of variables")
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(checkpoint):
            saver.restore(sess,checkpoint)
        else:
            print("Checkpoint of the model doesn't exist!")
            sys.exit()
        
        total_rew_eval = []
        epsilon = 0.05
        first_actions = {}
        for _ in np.arange(num_trials):
            s0 = preprocess_frame(env.reset(),shape_of_frame)
            zero_pad = np.zeros(s0.shape)
            if num_recent_obs == 1:
                sequence = []
            else:
                sequence = [zero_pad]*(num_recent_obs-1)
            sequence.append(s0)

            rew_eval = 0
            for step in np.arange(1,max_steps):
                env.render()
                if step%action_repeat == 1:
                    #calculate q1 and get optimal action
                    if step == 1:
                        obs_input = preprocess_obs(sequence[:num_recent_obs])
                    q_values = sess.run(q,feed_dict={obs_ph:np.array([obs_input])})
                    max_action = np.argmax(q_values,axis=-1)
                    #epsilon greedy algo
                    dice = np.random.uniform()
                    if dice < epsilon:
                        action = np.random.randint(num_actions)
                    else:
                        action = max_action
                obs,rew,done,_ = env.step(action)
                rew_eval += rew
                sequence.append(preprocess_frame(obs,shape_of_frame))
                last_obs_input  = obs_input
                obs_input = preprocess_obs(sequence[step:step+num_recent_obs])
                if done:
                    total_rew_eval.append(rew_eval)
                    break
        print("Evaluating the agent:")
        total_rew_eval = np.array(total_rew_eval)
        print("Total rewards of trials have max-{}, average-{}, std-{}.".format(np.amax(total_rew_eval),\
                np.mean(total_rew_eval),np.std(total_rew_eval)))
        
class RL_replay(object):
    def __init__(self,env,config):
        self.config = config
        self.env = env

    def initialize_replay(self):
        cap = self.config.replay_start_size
        shape_frame = self.config.shape_of_frame
        env = self.env
        replay = []
        last_obs = preprocess_frame(env.reset(),shape_frame)
        if self.config.num_recent_obs == 1:
            sequence = []
        else:
            sequence = [np.zeros(shape_frame)]*(self.config.num_recent_obs-1)
        sequence.append(last_obs)
        last_stack_obs = preprocess_obs(sequence)
        self.obs_shape = last_stack_obs.shape
        step = 1
        average_steps = []
        while len(replay) < cap:
            if step < self.config.max_steps:
                if step%self.config.action_repeat == 1:
                    action = env.action_space.sample()
                obs,rew,done,_ = env.step(action)
                sequence.append(preprocess_frame(obs,shape_frame))
                curr_stack_obs = preprocess_obs(sequence[step:step+self.config.num_recent_obs])
                replay.append((last_stack_obs,action,rew,curr_stack_obs,done))
                if done:
                    last_obs = preprocess_frame(env.reset(),shape_frame)
                    sequence = [np.zeros(shape_frame)]*(self.config.num_recent_obs-1)
                    sequence.append(last_obs)
                    last_stack_obs = preprocess_obs(sequence)
                    average_steps.append(step)
                    step = 0
                else:
                    last_stack_obs = curr_stack_obs
            else:
                last_obs = preprocess_frame(env.reset(),shape_frame)
                sequence = [np.zeros(shape_frame)]*(self.config.num_recent_obs-1)
                sequence.append(last_obs)
                last_stack_obs = preprocess_obs(sequence)
                step = 0
            step+=1
        average_steps = np.array(average_steps).astype(float)
        print("Number of steps taken to finish the game: average-{}, std-{}.".format(np.mean(average_steps),np.std(average_steps)))
        replay = np.array(replay)
        perm = np.random.permutation(cap)
        self.replay = replay[perm]

    def update_replay(self,new_samples):
        replay = np.concatenate([self.replay,np.array(new_samples)])
        size_replay = len(replay)
        if size_replay > self.config.replay_capacity:
            self.replay = self.replay[(size_replay-self.config.replay_capacity):]

    def sample_from_replay(self):
        idxes = np.random.randint(len(self.replay),size=self.config.batch)
        return self.replay[idxes]

class Prioritized_replay(RL_replay):
    def __init__(self,env,config):
        self.env = env
        self.config = config
        self.frequency_to_update_priorities_globally = self.config.replay_capacity*0.25

    def initialize_priority_list(self,priorities):
        priority_list = []
        for idx,priority in enumerate(priorities):
            priority_list.append([idx,priority])
        self.priority_list = np.array(priority_list)
    
    #priority_list is a list of tuples in the form of (idx,priority), sorted by priority, with idx pointing to the corresponding item in replay.
    #group_rbs is a list with length batch-1, which contains idxes that separate the priority_list into #batch groups.
    #The i-th group has members with priorities in [ptotal/batch*(i-1),ptotal/batch*i]
    def compute_priority_list(self,priority_list):
        batch = self.config.batch
        priority_list = np.array(priority_list)
        priority_list.sort(key=lambda x: x[1])
        ptotal = np.sum(priority_list[:,1])
        edges = [ptotal/batch*i for i in range(1,batch)]
        group_rbs = []
        lb = 0
        for edge in edges:
            rb = len(priority_list)-1
            if edge < priority_list[lb][1]:
                group_rbs.append(lb)
                continue
            elif edge > priority_list[rb][1]:
                group_rbs.append(rb)
                continue
            while (rb-lb)>1:
                medium = lb+int((rb-lb)/2)
                if edge < priority_list[medium][1]:
                    rb = medium
                elif edge > priority_list[medium][1]:
                    lb = medium
                else:
                    lb = medium
                    break
            group_rbs.append(lb)
        self.ptotal = ptotal
        self.group_rbs = np.array(group_rbs)
        self.priority_list = priority_list

    #if size of replay is smaller than replay_capacity, random sampling will be used and priority is not ordered.
    #once size of replay reaches replay_capacity, old experiences in the replay that have the closest priorities to 
    #new samples will be replaced by new samples.
    def update_replay(self,new_samples,priorities):
        old_size = len(self.replay)
        if old_size < self.config.replay_capacity:
            self.replay = np.concatenate([self.replay,np.array(new_samples)])
            new_size = len(self.replay)
            new_list = np.stack((np.arange(old_size,new_size),np.array(priorities)),axis=1)
            self.priority_list = np.concatenate([self.priority_list,new_list])
        else:
            batch = self.config.batch
            edges = [self.ptotal/batch*i for i in range(1,batch)]
            group_idxes = []
            #Figure out which groups new samples are in 
            for sample_idx,priority in enumerate(priorities):
                if priority < edges[0]:
                    group_idxes.append(0)
                    continue
                if priority > edges[-1]:
                    group_idxes.append(batch-1)
                    continue
                lb = 0; rb = len(edges)-1
                while (rb-lb)>1:
                    medium = lb+int((rb-lb)/2)
                    if priority < edges[medium][1]:
                        rb = medium
                    elif priority > edges[medium][1]:
                        lb = medium
                    else:
                        rb = medium+1
                        break
                group_idxes.append(rb)
            ptotal = self.ptotal
            #Get boundaries of groups that are in group_idxes
            for sample_idx,group_idx in enumerate(group_idxes):
                if group_idx == 0:
                    lb_idx = 0
                else:
                    lb_idx = self.group_rbs[group_idx-1]

                if group_idx == batch-1:
                    rb_idx = len(self.priority_list)-1
                else:
                    rb_idx = self.group_rbs[group_idx]

                #Get the closest idx in priority_list
                if lb_idx == rb_idx:
                    target_idx = rb_idx
                else:
                    trun_p_list = self.priority_list[lb_idx:rb_idx]
                    priority = priorities[sample_idx]
                    lb = 0; rb = len(trun_p_list) -1
                    while (rb-lb)>1:
                        medium = lb+int((rb-lb)/2)
                        if priority < trun_p_list[medium][1]:
                            rb = medium
                        elif priority > trun_p_list[medium][1]:
                            lb = medium
                        else:
                            lb = medium
                            break
                    target_idx = lb_idx+lb
                #Replace old experience with new one
                self.replay[self.priority_list[target_idx][0]] = new_samples[sample_idx]
                self.priority_list[target_idx][1] = priorities[sample_idx]

    #if size of replay is smaller than replay_capacity, random sampling will be used.
    #once size of replay reaches replay_capacity, prioritized sampling will be used.
    def sample_from_replay(self):
        if len(self.replay) < self.config.replay_capacity:
            idxes = np.random.randint(len(self.replay),size=self.config.batch)
            return self.replay[idxes],None
        else:
            samples = []
            sample_p_idxes = []
            lb = 0
            for rb in self.group_rbs+[len(self.replay)-1]:
                if rb == lb:
                    p_list_idx = lb
                else:
                    p_list_idx = np.random.randint(lb,high=rb)
                samples.append(self.replay[self.priority_list[p_list_idx][0]])
                sample_p_idxes.append(p_list_idx)
                lb = rb
            return np.array(samples),np.array(sample_p_idxes)

    def update_priority_after_sampling(self,p_indexed_priorities):
        self.priority_list[p_indexed_priorities[:,0]][1] = p_indexed_priorities[:,1]

            



