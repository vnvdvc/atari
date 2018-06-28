import gym,os,sys,subprocess,time
from skimage.transform import resize
from skimage.color import rgb2gray
from tensorflow.python.saved_model.simple_save import simple_save
import tensorflow as tf
import numpy as np
from rl_utils_prioritized import *


#prioritized experience replay, Tom Schaul et. al., 2016
class RL_config(object):
    shape_of_frame = (84,84)
    gamma = 0.99
    learning_rate = 0.0000625
    replay_capacity = 1000000
    replay_start_size = 50000
    num_recent_obs = 4
    action_repeat = 4
    steps_for_updating_q1 = 10000
    batch = 32
    max_episodes = 10000
    max_steps = 2000
    final_exploration_episodes = 5000
    episodes_to_save = 1000
    episodes_to_validate = 2000
    evaluation_trials = 30
    alpha_prioritized_replay = 0.6
    beta_prioritized_replay = 0.4


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
    rl_model = Prioritized_replay(env,rl_conf)
    rl_model.initialize_replay()
    obs_shape = rl_model.obs_shape
    print("Shape of the input:{}".format(obs_shape))
    if len(obs_shape) == 3:
        obs_h,obs_w,obs_c = rl_model.obs_shape
        obs_ph = tf.placeholder(tf.float32,shape=(None,obs_h,obs_w,obs_c),name="obs_ph")
        obs_dim = 3
    elif obs_shape < 3:
        if len(obs_shape) == 1:
            obs_h = obs_shape[0]
            obs_c = 1
        else:
            obs_h,obs_c = obs_shape
        obs_ph = tf.placeholder(tf.float32,shape=(None,obs_h,obs_c),name="obs_ph")
        obs_dim = 2
    else:
        print("obs_shape inconsistent with the model: {}".format(obs_shape))
        sys.exit()
    indexed_action_ph = tf.placeholder(tf.int32,shape=(None,2),name="indexed_action_ph")
    y_ph = tf.placeholder(tf.float32,shape=(None,1),name="y_ph")
    #ph for weights of importance sampling (IS)
    is_weight_ph = tf.placeholder(tf.float32,shape=None,name="is_weight_ph")


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
    q = tf.add(tf.matmul(a4,wo),bo,name="q")
    print("Shape of q: {}".format(q.get_shape().as_list()))

    #huber loss
    preds = tf.reshape(tf.gather_nd(q,indexed_action_ph),[-1,1],name="preds")
    loss = tf.losses.huber_loss(y_ph,preds,weights=is_weight_ph)
    reg_loss = loss + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    #Adam Optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=rl_conf.learning_rate).minimize(reg_loss)


    variable_set = [w1,b1,w2,b2,w3,b3,w4,b4,wo,bo]
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(checkpoint):
            saver.restore(sess,checkpoint)
            print("Restore model from checkpoint.")
        else:
            sess.run(tf.global_variables_initializer())
            print("Files saved to {}".format(save_dir))
        #Save model state
        print("Save model to save_dir:")
        inputs_dict = {
                "obs_ph": obs_ph,
                "indexed_action_ph": indexed_action_ph,
                "y_ph": y_ph,
                "is_weight_ph": is_weight_ph
                }
        outputs_dict = {
                "preds": preds
                }
        model_dir = os.path.join(save_dir,"model")
        if os.path.isdir(model_dir):
            model_dir = os.path.join(save_dir,"model-0")
            os.makedirs(model_dir)
        simple_save(sess,model_dir,inputs_dict,outputs_dict)

        q_file = os.path.join(save_dir,"q_values.txt")
        reward_file = os.path.join(save_dir,"rewards.txt")

        #compute priorites for initial replay
        batch_idx = 0
        priorities = np.zeros((len(rl_model.replay),1))
        variable_values_q1 = sess.run(variable_set)
        var_feed_dict_list_q1 = [(key,val) for key,val in zip(variable_set,variable_values_q1)]
        while batch_idx < len(rl_model.replay)/rl_conf.batch:
            if (batch_idx+1)*rl_conf.batch > len(rl_model.replay):
                samples = rl_model.replay[batch_idx*rl_conf.batch:]
            else:
                samples = rl_model.replay[batch_idx*rl_conf.batch:(batch_idx+1)*rl_conf.batch]
            sample_size = len(samples)
            obs_samples = np.array([sample[0] for sample in samples])
            obs_next_samples = np.array([sample[3] for sample in samples])
            rew_samples = np.array([sample[2] for sample in samples]).reshape((sample_size,1))
            done_samples = np.array([sample[4] for sample in samples]).reshape((sample_size,1))
            action_samples = np.array([sample[1] for sample in samples]).astype(int)

            q_targets = sess.run(q,feed_dict={obs_ph:obs_next_samples})

            q_for_priorities = sess.run(q,feed_dict={obs_ph:obs_samples})

            q_targets_selected_actions = np.amax(q_targets,axis=-1,keepdims=True)
            labels = np.where(done_samples,rew_samples,rew_samples+rl_conf.gamma*q_targets_selected_actions)

            if (batch_idx+1)*rl_conf.batch > len(rl_model.replay):
                q_priorities_selected_actions = q_for_priorities[np.arange(len(action_samples)),action_samples].reshape((sample_size,1))
                priorities[batch_idx*rl_conf.batch:] = np.absolute(labels - q_priorities_selected_actions)**(rl_conf.alpha_prioritized_replay)
            else:
                q_priorities_selected_actions = q_for_priorities[np.arange(len(action_samples)),action_samples].reshape((sample_size,1))
                priorities[batch_idx*rl_conf.batch:(batch_idx+1)*rl_conf.batch] = np.absolute(labels - q_priorities_selected_actions)**(rl_conf.alpha_prioritized_replay)
            batch_idx += 1
        rl_model.initialize_priority_list(priorities.flatten())


        update_q1_steps = 0
        for episode in np.arange(1,rl_conf.max_episodes+1):
            #Make sure the running time does not exceed walltime
            start = time.time()
            if start - time0 > float(walltime)*0.9:
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
            epsilon = max(1.0-float(episode)/rl_conf.final_exploration_episodes,0.1)
            #beta_prioritized_replay schedule
            beta_prioritized_replay = min(rl_conf.beta_prioritized_replay + float(episode)/rl_conf.final_exploration_episodes,1.0)
            for step in np.arange(1,rl_conf.max_steps):
                #whether to update variables of q1
                if update_q1_steps % rl_conf.steps_for_updating_q1 == 0:
                    variable_values_q1 = sess.run(variable_set)
                    var_feed_dict_list_q1 = [(key,val) for key,val in zip(variable_set,variable_values_q1)]
                #whether to recompute priority list
                if len(rl_model.replay) >= rl_conf.replay_capacity and update_q1_steps%rl_model.frequency_to_update_priorities_globally == 0:  
                    rl_model.compute_priority_list(rl_model.priority_list)
                if len(rl_model.replay) >= rl_conf.replay_capacity and rl_model.group_rbs == []:
                    rl_model.compute_priority_list(rl_model.priority_list)
                update_q1_steps += 1
                #double dqn
                samples,sample_p_idxes = rl_model.sample_from_replay()
                obs_samples = np.array([sample[0] for sample in samples])
                obs_next_samples = np.array([sample[3] for sample in samples])
                rew_samples = np.array([sample[2] for sample in samples]).reshape((rl_conf.batch,1))
                done_samples = np.array([sample[4] for sample in samples]).reshape((rl_conf.batch,1))
                action_samples = np.array([sample[1] for sample in samples]).astype(int)
                indexed_action_samples = np.array([[idx,sample[1]] for idx,sample in enumerate(samples)]).astype(int)

                q_for_selecting_actions = sess.run(q,feed_dict={obs_ph:obs_next_samples})
                selected_actions = np.argmax(q_for_selecting_actions,axis=-1)

                q_for_priorities = sess.run(q,feed_dict={obs_ph:obs_samples})

                q_targets = sess.run(q,feed_dict=dict([(obs_ph,obs_next_samples)]+var_feed_dict_list_q1))
                q_targets_selected_actions = q_targets[np.arange(len(selected_actions)),selected_actions].reshape((rl_conf.batch,1))
                labels = np.where(done_samples,rew_samples,rew_samples+rl_conf.gamma*q_targets_selected_actions)

                #update priorities for replay
                if sample_p_idxes is not None:
                    priorities = np.absolute(labels - q_for_priorities[np.arange(len(action_samples)),action_samples].reshape((rl_conf.batch,1)))**(rl_conf.alpha_prioritized_replay)
                    p_indexed_priorities = np.concatenate((sample_p_idxes.reshape((rl_conf.batch,1)),priorities),axis=1)
                    rl_model.update_priority_after_sampling(p_indexed_priorities)
                    weights = (len(rl_model.replay)*priorities)**(-beta_prioritized_replay)
                    weights = weights/np.amax(weights)
                else:
                    weights = 1.0

                sess.run(train_op,feed_dict={obs_ph:obs_samples,y_ph:labels,indexed_action_ph:indexed_action_samples,is_weight_ph:weights})

                
                #update replay
                if step%rl_conf.action_repeat == 1:
                    #calculate q1 and get optimal action
                    if step == 1:
                        obs_input = preprocess_obs(sequence[:rl_conf.num_recent_obs])
                    q_values_for_new_sample = sess.run(q,feed_dict={obs_ph:np.array([obs_input])})
                    max_action_for_new_sample = np.argmax(q_values_for_new_sample,axis=-1)
                    average_q += q_values_for_new_sample[0][max_action_for_new_sample]
                    #epsilon greedy algo
                    dice = np.random.uniform()
                    if dice < epsilon:
                        action = np.random.randint(num_actions)
                    else:
                        action = max_action_for_new_sample
                obs,rew,done,_ = env.step(action)
                total_rew += rew
                sequence.append(preprocess_frame(obs,rl_conf.shape_of_frame))
                last_obs_input  = obs_input
                obs_input = preprocess_obs(sequence[step:step+rl_conf.num_recent_obs])
                #calculate priority for new sample
                if done:
                    priority = np.absolute(rew - q_values_for_new_sample[0][action])**(rl_conf.alpha_prioritized_replay)
                else:
                    q_target_for_new_sample =  sess.run(q,feed_dict=dict([(obs_ph,np.array([obs_input]))]+var_feed_dict_list_q1))
                    priority = np.absolute(rew + rl_conf.gamma*q_target_for_new_sample[0][max_action_for_new_sample] - q_values_for_new_sample[0][action]\
                            )**(rl_conf.alpha_prioritized_replay)
                if np.isscalar(priority):
                    priority = np.array([priority])
                else:
                    priority = priority.flatten()
                rl_model.update_replay(np.array([(last_obs_input,action,rew,obs_input,done)]),priority)
                if done:
                    with open(q_file,"a+") as out:
                        out.write(str(average_q*4/step)+"\n")
                    with open(reward_file,"a+") as out:
                        out.write(str(total_rew)+"\n")
                    break
            if episode%rl_conf.episodes_to_save == 0:
                saver.save(sess,checkpoint)
                save_file_to_distant_dir = """cp {} {}""".format(os.path.join(save_dir,"*"),distant_dir)
                subprocess.call(save_file_to_distant_dir,shell=True)
            print("Time taken to complete this episode: {}s.".format(time.time()-start))
            
            #Evaluate the performance of the agent with total rewards
            if episode%rl_conf.episodes_to_validate == 0:
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
                            q_values_for_evaluation = sess.run(q,feed_dict={obs_ph:np.array([obs_input])})
                            max_action_for_evaluation = np.argmax(q_values_for_evaluation,axis=-1)
                            #epsilon greedy algo
                            dice = np.random.uniform()
                            if dice < epsilon:
                                action = np.random.randint(num_actions)
                            else:
                                action = max_action_for_evaluation
                        obs,rew,done,_ = env.step(action)
                        rew_eval += rew
                        sequence.append(preprocess_frame(obs,rl_conf.shape_of_frame))
                        last_obs_input  = obs_input
                        obs_input = preprocess_obs(sequence[step:step+rl_conf.num_recent_obs])
                        if done:
                            total_rew_eval.append(rew_eval)
                            break
                print("Evaluating the agent at episode-{}:".format(episode))
                total_rew_eval = np.array(total_rew_eval)
                print("Total rewards of trials have max-{}, average-{}, std-{}.".format(np.amax(total_rew_eval),\
                        np.mean(total_rew_eval),np.std(total_rew_eval)))

    


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




