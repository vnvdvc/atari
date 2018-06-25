import gym,os,sys,subprocess,time
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

class RL_config(object):
    shape_of_frame = (84,84)
    gamma = 0.99
    learning_rate = 0.00025
    replay_capacity = 1000000
    replay_start_size = 10000
    num_recent_obs = 4
    action_repeat = 4
    steps_for_updating_q1 = 10000
    batch = 32
    max_episodes = 1000
    max_steps = 2000
    final_exploration_frame = 1000000
    episodes_to_save = 100
    steps_to_validate = 250000
    evaluation_trials = 30

class RL_model(object):
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

def main(save_dir,distant_dir,walltime):
    """
    Atari games have two kinds of inputs,
    ram: size (128,)
    human: size (screen_height,screen_width,3)
    Since the preprocessing stacks several frames (RL_config.num_recent_obs) together,
    the input size should be modified accordingly.
    """
    game = "SpaceInvaders-v0"
    print("Playing game {}".format(game.split("-")[0]))
    env = gym.make(game)
    checkpoint = os.path.join(save_dir,game+"-dqn.ckpt")
    time0 = time.time()

    num_actions = env.action_space.n
    print("Number of actions is {}".format(num_actions))
    rl_conf = RL_config()
    rl_model = RL_model(env,rl_conf)
    rl_model.initialize_replay()
    obs_shape = rl_model.obs_shape
    print("Shape of the input:{}".format(obs_shape))
    if len(obs_shape) == 3:
        obs_h,obs_w,obs_c = rl_model.obs_shape
        obs_ph = tf.placeholder(tf.float32,shape=(None,obs_h,obs_w,obs_c))
        obs_dim = 3
    elif obs_shape < 3:
        if len(obs_shape) == 1:
            obs_h = obs_shape[0]
            obs_c = 1
        else:
            obs_h,obs_c = obs_shape
        obs_ph = tf.placeholder(tf.float32,shape=(None,obs_h,obs_c))
        obs_dim = 2
    else:
        print("obs_shape inconsistent with the model: {}".format(obs_shape))
        sys.exit()
    indexed_action_ph = tf.placeholder(tf.int32,shape=(None,2))
    y_ph = tf.placeholder(tf.float32,shape=(None,1))


    #3 cnn layers
    if obs_dim == 3:
        w1 = tf.get_variable("cnn_w1",[8,8,obs_c,32],dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("cnn_w2",[4,4,32,64],dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer())
        w3 = tf.get_variable("cnn_w3",[3,3,64,64],dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer())
    else:
        w1 = tf.get_variable("cnn_w1",[8,obs_c,32],dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("cnn_w2",[4,32,64],dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer())
        w3 = tf.get_variable("cnn_w3",[3,64,64],dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer())


    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,1/2*tf.norm(w1))
    if obs_dim == 3:
        z1 = tf.nn.conv2d(obs_ph,w1,strides=[1,4,4,1],padding="VALID")
    else:
        z1 = tf.nn.conv1d(obs_ph,w1,strides=[1,4,1],padding="VALID")
    b1 = tf.get_variable("cnn_b1",z1.get_shape().as_list()[1:],dtype=tf.float32,\
            initializer=tf.zeros_initializer())
    a1 = tf.nn.relu(z1+b1)


    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,1/2*tf.norm(w2))
    if obs_dim == 3:
        z2 = tf.nn.conv2d(a1,w2,strides=[1,2,2,1],padding="VALID")
    else:
        z2 = tf.nn.conv1d(a1,w2,strides=[1,2,1],padding="VALID")
    b2 = tf.get_variable("cnn_b2",z2.get_shape().as_list()[1:],dtype=tf.float32,\
            initializer=tf.zeros_initializer())
    a2 = tf.nn.relu(z2+b2)

    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,1/2*tf.norm(w3))
    if obs_dim == 3:
        z3 = tf.nn.conv2d(a2,w3,strides=[1,1,1,1],padding="VALID")
    else:
        z3 = tf.nn.conv1d(a2,w3,strides=[1,1,1],padding="VALID")
    b3 = tf.get_variable("cnn_b3",z3.get_shape().as_list()[1:],dtype=tf.float32,\
            initializer=tf.zeros_initializer())
    a3 = tf.nn.relu(z3+b3)

    #fully connected relu
    a3_flat = tf.contrib.layers.flatten(a3)
    w4 = tf.get_variable("fc_w4",[a3_flat.get_shape().as_list()[1],512],dtype=tf.float32,\
            initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,1/2*tf.norm(w4))
    b4 = tf.get_variable("fc_b4",[1,512],dtype=tf.float32,\
            initializer=tf.zeros_initializer())
    a4 = tf.nn.relu(tf.matmul(a3_flat,w4)+b4)

    #linear output layer
    wo = tf.get_variable("wo",[512,num_actions],dtype=tf.float32,\
            initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,1/2*tf.norm(wo))
    bo = tf.get_variable("bo",[1,num_actions],dtype=tf.float32,\
            initializer=tf.zeros_initializer())
    #q is an array with the size of num_actions
    q = tf.matmul(a4,wo)+bo
    print("Shape of q: {}".format(q.get_shape().as_list()))

    #huber loss
    preds = tf.reshape(tf.gather_nd(q,indexed_action_ph),[-1,1])
    loss = tf.losses.huber_loss(y_ph,preds)
    reg_loss = loss + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    #Adam Optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=rl_conf.learning_rate).minimize(reg_loss)

    variable_set = [w1,b1,w2,b2,w3,b3,w4,b4,wo,bo]
    sys.exit()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(checkpoint):
            saver.restore(sess,checkpoint)
            print("Restore model from checkpoint.")
        else:
            sess.run(tf.global_variables_initializer())
            print("Files saved to {}".format(save_dir))
        q_file = os.path.join(save_dir,"q_values.txt")
        reward_file = os.path.join(save_dir,"rewards.txt")

        update_q1_steps = 0
        for episode in np.arange(1,rl_conf.max_episodes+1):
            #Make sure the running time does not exceed walltime
            start = time.time()
            if start - time0 > float(walltime)*0.8:
                saver.save(sess,checkpoint)
                print("Running time limit achieves.")
                sys.exit()
            #metrics
            total_rew  = 0.0
            average_q = 0.0

            s0 = preprocess_frame(env.reset(),rl_conf.shape_of_frame)
            zero_pad = np.zeros(s0.shape)
            if rl_conf.num_recent_obs == 1:
                sequence = []
            else:
                sequence = [zero_pad]*(rl_conf.num_recent_obs-1)
            sequence.append(s0)
            #epsilon schedule
            epsilon = max(1.0-float(episode)/rl_conf.final_exploration_frame,0.1)
            for step in np.arange(1,rl_conf.max_steps):
                    #whether to update variables of q1
                if update_q1_steps % rl_conf.steps_for_updating_q1 == 0:
                    variable_values_q1 = sess.run(variable_set)
                    var_feed_dict_list_q1 = [(key,val) for key,val in zip(variable_set,variable_values_q1)]
                update_q1_steps += 1
                #one step of SGD
                samples = rl_model.sample_from_replay()
                obs_samples = np.array([sample[0] for sample in samples])
                rew_samples = np.array([sample[2] for sample in samples]).reshape((rl_conf.batch,1))
                done_samples = np.array([sample[4] for sample in samples]).reshape((rl_conf.batch,1))
                indexed_action_samples = np.array([[idx,sample[1]] for idx,sample in enumerate(samples)])
                q_targets = sess.run(q,feed_dict=dict([(obs_ph,obs_samples)]+var_feed_dict_list_q1))
                labels = np.where(done_samples,rew_samples,rew_samples+rl_conf.gamma*np.amax(q_targets,axis=-1,keepdims=True))
                sess.run(train_op,feed_dict={obs_ph:obs_samples,y_ph:labels,indexed_action_ph:indexed_action_samples})
                
                #update replay
                if step%rl_conf.action_repeat == 1:
                    #calculate q1 and get optimal action
                    if step == 1:
                        obs_input = preprocess_obs(sequence[:rl_conf.num_recent_obs])
                    q_values = sess.run(q,feed_dict={obs_ph:np.array([obs_input])})
                    max_action = np.argmax(q_values,axis=-1)
                    average_q += q_values[0][max_action]
                    #epsilon greedy algo
                    dice = np.random.uniform()
                    if dice < epsilon:
                        action = np.random.randint(num_actions)
                    else:
                        action = max_action
                obs,rew,done,_ = env.step(action)
                total_rew += rew
                sequence.append(preprocess_frame(obs,rl_conf.shape_of_frame))
                last_obs_input  = obs_input
                obs_input = preprocess_obs(sequence[step:step+rl_conf.num_recent_obs])
                rl_model.update_replay(np.array([(last_obs_input,action,rew,obs_input,done)]))
                if done:
                    with open(q_file,"a+") as out:
                        out.write(str(average_q*4/step)+"\n")
                    with open(reward_file,"a+") as out:
                        out.write(str(total_rew)+"\n")
                    break
            if episode%rl_conf.episodes_to_save == 0:
                saver.save(sess,checkpoint)
                save_file_to_distant_dir = """cp {} {}""".format(q_file,os.path.join(distant_dir,"q_values.txt"))
                subprocess.call(save_file_to_distant_dir,shell=True)
            print("Time taken to complete this episode: {}s.".format(time.time()-start))
            
            #Evaluate the performance of the agent with total rewards
            if update_q1_steps%rl_conf.steps_to_validate == 0:
                total_rew_eval = []
                epsilon = 0.05
                first_actions = {}
                for _ in np.arange(rl_conf.evaluation_trials):
                    s0 = preprocess_frame(env.reset(),rl_conf.shape_of_frame)
                    zero_pad = np.zeros(s0.shape)
                    if rl_conf.num_recent_obs == 1:
                        sequence = []
                    else:
                        sequence = [zero_pad]*(rl_conf.num_recent_obs-1)
                    sequence.append(s0)

                    rew_eval = 0
                    for step in np.arange(1,rl_conf.max_steps):
                        if step%rl_conf.action_repeat == 1:
                            #calculate q1 and get optimal action
                            if step == 1:
                                obs_input = preprocess_obs(sequence[:rl_conf.num_recent_obs])
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
                        sequence.append(preprocess_frame(obs,rl_conf.shape_of_frame))
                        last_obs_input  = obs_input
                        obs_input = preprocess_obs(sequence[step:step+rl_conf.num_recent_obs])
                        if done:
                            total_rew_eval.append(rew_eval)
                            break
                print("Evaluating the agent at step-{}:".format(update_q1_steps))
                total_rew_eval = np.array(total_rew_eval)
                print("Total rewards of trials have average-{}, std-{}.".format(np.mean(total_rew_eval),np.std(total_rew_eval)))

    


if __name__ == "__main__":
    argvs = sys.argv[1:]
    save_dir = argvs[0]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    distant_dir = argvs[1]
    if not os.path.isdir(distant_dir):
        os.makedirs(distant_dir)
    walltime = argvs[2]
    main(save_dir,distant_dir,walltime)




