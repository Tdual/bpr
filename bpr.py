
# coding: utf-8

# data from https://grouplens.org/datasets/movielens/
# 

# In[1]:

import numpy as np
import tensorflow as tf
from collections import defaultdict
import gensim as gs
try:
    # noinspection PyUnresolvedReferences
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        from tqdm import tqdm_notebook as tqdm
    else:
        raise RuntimeError
except (NameError, RuntimeError):
    from tqdm import tqdm


# In[2]:

def load_data(data_path):
    data = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open(data_path, 'r') as f:
        f.readline()
        for idx, line in enumerate(f):
            u, i, _, _ = line.split(",")
            u = int(u)
            i = int(i)
            data[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
            if idx == 1000:
                break
    return max_u_id, max_i_id, data


# In[3]:

def map_data(data_path):
    line_list =[]
    user_list = []
    item_dic = {}
    few_buyers =[]
    data = defaultdict(set)
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line[:-1] # remove \n
            l = line.split(",")
            user_id = l[0]
            items = l[1:]
            user_list.append(user_id)
            line_list.append(items)
    dictionary = gs.corpora.Dictionary(line_list)
    for u, items in zip(user_list, line_list):
        data[u].update([dictionary.token2id[item] for item in items])
    for u,i in data.items():
        if len(i) < 10:
            few_buyers.append(u)
    for u in few_buyers:
        del data[u]
    d = {}
    user_list = []
    for idx,(u,i) in enumerate(data.items()):
        d[idx] = i
        user_list.append(u)
    user_count = len(data.keys())
    item_count = len(dictionary)
    return (user_count, item_count, d)


# In[4]:

user_count, item_count, data = map_data("./data.csv")


# In[5]:

#data


# In[6]:

print("item count: ", item_count)
print("user count: ", user_count)


# In[ ]:




# In[7]:

def generate_test(data):
    user_test = dict()
    for u, i_list in data.items():
        user_test[u] = np.random.choice(list(i_list))
    return user_test


# In[8]:

#data_path = "./ml-20m/ratings.csv"
#user_count, item_count, data = load_data(data_path)
user_ratings_test = generate_test(data)


# In[9]:

def generate_train_batch(data, user_ratings_test, item_count, batch_size=512):
    t = []
    for _ in range(batch_size):
        u = np.random.choice(list(data.keys()))
        i = np.random.choice(list(data[u]))
        while i == user_ratings_test[u]:
            i = np.random.choice(list(data[u]))
        
        j = np.random.randint(1, item_count+1)
        while j in data[u]:
            j = np.random.randint(1, item_count+1)
        t.append([u, i, j])
    return np.asarray(t)

def generate_test_batch(user_ratings, user_ratings_test, item_count):
    for u in np.random.choice(list(user_ratings.keys()),300):
        t = []
        i = user_ratings_test[u]
        for j in range(1, item_count+1):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield np.asarray(t)


# In[ ]:




# In[10]:

def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))

def bias_variable(shape):
    return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))


# In[11]:

def bpr(user_count, item_count, hidden_dim, batch_size=512):
    
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    user_w = weight_variable([user_count+1, hidden_dim])
    item_w = weight_variable([item_count+1, hidden_dim])
    item_b = bias_variable([item_count+1, 1])
        
        
    u_e = tf.nn.embedding_lookup(user_w, u)
        
    i_e = tf.nn.embedding_lookup(item_w, i)
    i_b = tf.nn.embedding_lookup(item_b, i)
        
    j_e = tf.nn.embedding_lookup(item_w, j)
    j_b = tf.nn.embedding_lookup(item_b, j)
    
    # MF 
    x = i_b - j_b + tf.reduce_sum(tf.matmul(u_e, tf.transpose((i_e - j_e))), 1, keep_dims=True)
    
    
    auc_per_user = tf.reduce_mean(tf.cast(x > 0,"float"))
    
    l2_norm = tf.add_n([
            tf.reduce_sum(tf.norm(u_e)), 
            tf.reduce_sum(tf.norm(i_e)),
            tf.reduce_sum(tf.norm(j_e))
        ])
    
    regu_rate = 0.0001
    loss = - tf.reduce_mean(tf.log(tf.sigmoid(x))) + regu_rate * l2_norm
    
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    return u, i, j, auc_per_user, loss, train_op


# In[ ]:

with tf.Session() as session:
    u, i, j, auc_per_user, loss, train_op = bpr(user_count, item_count, 20)
    session.run(tf.global_variables_initializer())
    for epoch in range(10):
        _batch_loss = 0
        for index in tqdm(range(2000)): 
            uij = generate_train_batch(data, user_ratings_test, item_count)
            _loss, _ = session.run([loss, train_op], feed_dict={u:uij[:,0], i:uij[:,1], j:uij[:,2]})
            _batch_loss += _loss
                   
        print("epoch: ", epoch, ", loss: ", _batch_loss / (index+1))


        _auc_sum = 0.0
        user_count = 0
        for t_uij in tqdm(generate_test_batch(data, user_ratings_test, item_count)):
            _auc_per_user, _test_loss = session.run([auc_per_user, loss],feed_dict={u:t_uij[:,0], i:t_uij[:,1], j:t_uij[:,2]})
            user_count += 1
            _auc_sum += _auc_per_user
            
            _auc = _auc_sum/user_count # eq (1) in the paper
            
        print("test loss: ", _test_loss, ", test auc: ", _auc)


# In[ ]:



