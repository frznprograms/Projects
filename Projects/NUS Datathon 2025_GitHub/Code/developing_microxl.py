# -*- coding: utf-8 -*-
"""developing_microxl.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rGMO8pAOv2yIoclRZgPBu5OXFGdQdm8t

# Imports #
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip uninstall -y tensorflow keras tensorflow-recommenders tensorflow-text tensorflow-tpu tf-keras
# %pip install -U tensorflow==2.17 keras==3.5.0 tensorflow-recommenders tensorflow-text

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q tensorflow-recommenders
# %pip install --upgrade tensorflow_ranking

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import tensorflow as tf
from scipy.sparse import coo_matrix
from typing import Dict, Text
from sklearn.utils import resample

import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

#drive.mount("/content/drive/")

"""# Data Preparation #

## Data Cleaning ##

### Preparing var_1 Data ###
"""

var_1_df = pd.read_parquet("")

var_1_df.columns

# drop columns to maintain ethical standards
var_1_df.drop(columns=[
    'race_desc_map',
    'cltpcode',
    'household_size',
    'family_size'
], axis=1, inplace=True)

cols_to_encode = ['marryd', 'cltsex', 'household_size_grp', 'family_size_grp']
for col in cols_to_encode:
    encoder = LabelEncoder()
    var_1_df[col] = encoder.fit_transform(var_1_df[col])

this_year = datetime.now().year
var_1_df['age'] = this_year - var_1_df['cltdob'].dt.year

var_1_df.drop(columns=[
    'cltdob'
], axis=1, inplace=True)

var_1_df['economic_status'] = pd.to_numeric(var_1_df['economic_status'], errors='coerce')
var_1_df['economic_status'] = var_1_df['economic_status'].fillna(var_1_df['economic_status'].median())
var_1_df['economic_status'] = var_1_df['economic_status'].astype('int32')

var_1_df['age'] = var_1_df['age'].fillna(var_1_df['age'].median()).astype('int32')

var_1_df['cltsex'] = var_1_df['cltsex'].astype('int32')
var_1_df['household_size_grp'] = var_1_df['household_size_grp'].astype('int32')
var_1_df['family_size_grp'] = var_1_df['family_size_grp'].astype('int32')
var_1_df['marryd'] = var_1_df['marryd'].astype('int32')

var_1_df['secuityno'] = var_1_df['secuityno'].apply(lambda string: string[4:])
var_1_df['secuityno'] = var_1_df['secuityno'].astype('int32')

"""### Preparing var_2 Data ###"""

var_2_df = pd.read_parquet("")

test = var_2_df[var_2_df['product'] == 'prod_6']
print(test['annual_premium'].mean())

# Maximum count (12082 for product 8.0)
max_count = var_2_df['product'].value_counts().max()

# Initialize an empty list to hold the resampled data
oversampled_df = []

# Loop over each unique product and resample
for product, count in var_2_df['product'].value_counts().items():
    product_df = var_2_df[var_2_df['product'] == product]
    if count < max_count:
        # Resample the product to match the max_count
        product_df_oversampled = resample(
            product_df, replace=True, n_samples=max_count - count, random_state=42
        )
        oversampled_df.append(product_df_oversampled)

    oversampled_df.append(product_df)

# Concatenate the list of resampled dataframes back into one dataframe
oversampled_policy_df = pd.concat(oversampled_df)

policy_df = oversampled_policy_df

var_2_df['cust_tenure_at_purchase_grp'].value_counts()
encoder = LabelEncoder()
var_2_df['cust_tenure_at_purchase_grp'] = encoder.fit_transform(var_2_df['cust_tenure_at_purchase_grp'])

var_2_df['product_grp'].value_counts()
encoder = LabelEncoder()
var_2_df['product_grp'] = encoder.fit_transform(var_2_df['product_grp'])

len(var_2_df)

var_2_df.drop(columns=[
    'flg_main', 'flg_rider', 'flg_inforce',
    'flg_lapsed', 'flg_cancel', 'flg_expire',
    'flg_converted', 'occdate', 'chdrnum',
    'agntnum', 'cust_age_at_purchase_grp'
], axis=1, inplace=True)

var_2_df['product'] = var_2_df['product'].apply(lambda string: string[-1:])
var_2_df['product'] = var_2_df['product'].astype('int32')

prod_types = list(var_2_df['product'].unique())
mean_values = {}
for prod in prod_types:
  subset_data = var_2_df[var_2_df['product'] == prod]
  mean_value = subset_data['annual_premium'].mean()
  mean_values[prod] = mean_value

var_2_df['mean_premium'] = var_2_df['product'].map(mean_values)

var_2_df.drop(columns=['annual_premium'], axis=1, inplace=True)

var_2_df['secuityno'] = var_2_df['secuityno'].apply(lambda string: string[4:])
var_2_df['secuityno'] = var_2_df['secuityno'].astype('int32')

var_2_df['product'].unique()

"""### Preparing var_3 Data ###"""

var_3_df = pd.read_parquet("")

var_3_df.columns

var_3_df.drop(columns=[
    'agent_age', 'agent_gender', 'agent_marital', 'agent_tenure',
    'cnt_converted', 'annual_premium_cnvrt', 'pct_lapsed', 'pct_cancel',
    'pct_inforce', 'pct_SX0_unknown',
    'pct_SX1_male', 'pct_SX2_female', 'pct_AG01_lt20', 'pct_AG02_20to24',
    'pct_AG03_25to29', 'pct_AG04_30to34', 'pct_AG05_35to39',
    'pct_AG06_40to44', 'pct_AG07_45to49', 'pct_AG08_50to54',
    'pct_AG09_55to59', 'pct_AG10_60up',
], axis=1, inplace=True)

type(var_3_df['agent_product_expertise'].iloc[0][0])

def process_expertise(item):
    new_list = []
    for elem in item:
        new_list.append(int(elem[-1]))
    return new_list

var_3_df['agent_product_expertise'] = var_3_df['agent_product_expertise'].apply(lambda item: process_expertise(item))

"""### Summarizing the Data ###"""

var_2_df.columns

var_1_not_in_var_2 = var_1_df[~var_1_df['secuityno'].isin(var_2_df['secuityno'])]
print(len(var_1_not_in_var_2['secuityno']))

merged_data = pd.merge(var_1_df, var_2_df, on='secuityno', how='outer')

merged_data.head()

var_1_df = merged_data[['secuityno', 'marryd', 'cltsex', 'economic_status', 'household_size_grp',
       'family_size_grp', 'age', 'cust_tenure_at_purchase_grp']]

var_2_df = merged_data[['secuityno', 'product', 'product_grp', 'mean_premium']]

"""# Final Tensors and Inputs #

The matrix created in this step will serve as A, which can then be used for matrix factorization where A = UV^T.
"""

policy_df["mean_premium"] = (policy_df["mean_premium"] - policy_df["mean_premium"].mean()) / policy_df["mean_premium"].std()
var_1_df["age"] = (var_1_df["age"] - var_1_df["age"].mean()) / var_1_df["age"].std()

unique_clients = var_1_df["secuityno"].unique()
unique_policies = policy_df["product"].unique()

# Create mappings for client and policy IDs to matrix indices
var_1_to_index = {client: idx for idx, client in enumerate(unique_clients)}
policy_to_index = {policy: idx for idx, policy in enumerate(unique_policies)}
index_to_policy = {idx: policy for policy, idx in policy_to_index.items()}  # Reverse mapping

# Number of clients and policies
num_clients = len(unique_clients)
num_policies = len(unique_policies)

# Manually set correct feature sizes
client_feature_size = len(["marryd", "cltsex", "economic_status", "household_size_grp", "family_size_grp", "age", "cust_tenure_at_purchase_grp"])
policy_feature_size = len(["product_grp", "mean_premium"])

data = []
for client in unique_clients:
    client_id = client_to_index[client]
    policies_client_bought = policy_df[policy_df['secuityno'] == client]['product'].unique()
    policies_client_didnt_buy = [p for p in unique_policies if p not in policies_client_bought]

    client_features = client_df.loc[client_df['secuityno'] == client, [
        "marryd", "cltsex", "economic_status", "household_size_grp", "family_size_grp", "age", "cust_tenure_at_purchase_grp"
    ]].values.flatten().astype(np.float32)

    # Ensure fixed shape (truncate if too long, pad if too short)
    client_features = np.resize(client_features, client_feature_size)

    for policy in policies_client_bought:
        policy_id = policy_to_index[policy]
        policy_features = policy_df[policy_df['product'] == policy][["product_grp", "mean_premium"]].values.flatten().astype(np.float32)
        policy_features = np.resize(policy_features, policy_feature_size)
        data.append((client_id, policy_id, client_features, policy_features, 1))

    for policy in policies_client_didnt_buy:
        policy_id = policy_to_index[policy]
        policy_features = policy_df[policy_df['product'] == policy][["product_grp", "mean_premium"]].values.flatten().astype(np.float32)
        policy_features = np.resize(policy_features, policy_feature_size)
        data.append((client_id, policy_id, client_features, policy_features, 0))



# Convert data to TensorFlow dataset
def convert_to_tf_dataset(data):
    client_ids, policy_ids, client_features, policy_features, labels = zip(*data)

    # Ensure features are stacked properly to form rectangular arrays
    client_features = np.stack(client_features).astype(np.float32)
    policy_features = np.stack(policy_features).astype(np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Ensure labels are exactly 0 or 1
    print("Unique labels:", np.unique(labels))

    dataset = tf.data.Dataset.from_tensor_slices({
        "client_id": np.array(client_ids, dtype=np.int32),
        "policy_id": np.array(policy_ids, dtype=np.int32),
        "client_features": client_features,
        "policy_features": policy_features,
        "label": labels
    })
    return dataset

interaction_dataset = convert_to_tf_dataset(data)





"""# Model Development #

## Model Training ##
"""

np.random.seed(40)
tf.random.set_seed(40)

# Shuffle and split dataset
dataset_size = interaction_dataset.cardinality().numpy()  # Get the dataset size
train_size = int(0.8 * dataset_size)

shuffled = interaction_dataset.shuffle(dataset_size, seed=40, reshuffle_each_iteration=False)
train = shuffled.take(train_size)
test = shuffled.skip(train_size)

class ClientPolicyModeller(tf.keras.Model):
    def __init__(self, num_clients, num_policies, embedding_dim=64):
        super().__init__()

        # Reduce client ID embedding size
        self.client_embeddings = tf.keras.layers.Embedding(num_clients, embedding_dim // 2)
        self.policy_embeddings = tf.keras.layers.Embedding(num_policies, embedding_dim // 2)

        # Increase metadata feature embeddings
        self.client_dense = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.client_dropout = tf.keras.layers.Dropout(0.4)
        self.policy_dense = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.policy_dropout = tf.keras.layers.Dropout(0.4)

        # Final projection layer to align dimensions
        self.final_dense = tf.keras.layers.Dense(embedding_dim, activation='relu')

    def call(self, inputs: Dict[Text, tf.Tensor]):
        client = inputs["client_id"]
        policy = inputs["policy_id"]
        client_features = inputs["client_features"]
        policy_features = inputs["policy_features"]

        # Embed client ID and policy ID
        client_emb = self.client_embeddings(client)
        policy_emb = self.policy_embeddings(policy)

        # Process metadata features
        client_feat = self.client_dense(client_features)
        policy_feat = self.policy_dense(policy_features)

        # Concatenate embeddings and metadata features
        combined_client = tf.concat([client_emb, client_feat], axis=1)
        combined_policy = tf.concat([policy_emb, policy_feat], axis=1)

        # Final projection to align dimensions
        combined_client = self.final_dense(combined_client)
        combined_policy = self.final_dense(combined_policy)

        return tf.keras.activations.sigmoid(tf.reduce_sum(combined_client * combined_policy, axis=1, keepdims=True))

@tf.keras.saving.register_keras_serializable()
class MicroXL(tfrs.models.Model):
    def __init__(self, num_clients, num_policies):
        super().__init__()
        self.ranking_model = ClientPolicyModeller(num_clients, num_policies)
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            #metrics=[tfr.keras.metrics.NDCGMetric(name="ndcg")]
        )

    def build(self, input_shape):
        self.ranking_model.build(input_shape)

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(features)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features["label"]
        rating_predictions = self(features)
        return self.task(labels=labels, predictions=rating_predictions)

# Detect TPU and initialize
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)  # Use TPU strategy
    print("Running on TPU:", tpu.master())
except ValueError:
    strategy = tf.distribute.get_strategy()  # Default to CPU/GPU
    print("Running on CPU/GPU")

NUM_CLIENTS = client_df['secuityno'].nunique()
NUM_POLICIES = policy_df['product'].nunique()

NUM_CLIENTS, NUM_POLICIES

np.random.seed(40)
tf.random.set_seed(40)

with strategy.scope():  # Place model inside TPU scope
    model = MicroXL(num_clients=NUM_CLIENTS, num_policies=NUM_POLICIES)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))

# Train the model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
model.fit(train.batch(128), epochs=10  , callbacks=[callback])

model.save('XLMicro_v3.keras')

"""## Model Testing/Inference ##"""

# Function to recommend policies
def recommend_policies(model, client_id, client_features, top_k=2):
    policy_ids = np.array(list(policy_to_index.values()), dtype=np.int32)

    # Ensure all unique policies are retrieved
    policy_features = policy_df.groupby("product").first().reindex(unique_policies)[["product_grp", "mean_premium"]]
    policy_features = policy_features.fillna(0).to_numpy(dtype=np.float32)  # Handle any missing values

    client_ids = np.full((num_policies,), client_id, dtype=np.int32)
    client_features = np.tile(client_features.reshape(1, -1), (num_policies, 1))

    # Ensure correct shape for policy_features
    if policy_features.shape[0] != num_policies:
        raise ValueError(f"policy_features shape mismatch: Expected ({num_policies}, {policy_feature_size}), but got {policy_features.shape}")

    inputs = {
        "client_id": client_ids,
        "policy_id": policy_ids,
        "client_features": client_features.astype(np.float32),
        "policy_features": policy_features.astype(np.float32)  # Ensure proper shape
    }

    scores = model(inputs).numpy().flatten()
    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    recommended_policy_ids = [index_to_policy[idx] for idx in top_k_indices]  # Convert back to original policy IDs
    return recommended_policy_ids, scores[top_k_indices]



"""### Model Evaluation ###"""

# Function to evaluate model
def evaluate_model(model, test_data, top_k=2):
    correct_predictions = 0
    total_samples = 0

    for test_sample in test_data:
        client_id = test_sample["client_id"].numpy()
        true_policy = test_sample["policy_id"].numpy()
        client_features = test_sample["client_features"].numpy()

        # Convert true_policy to original if needed
        if true_policy in index_to_policy:
            true_policy = index_to_policy[true_policy]  # Ensure consistency with mapped policies

        recommended_policies, _ = recommend_policies(model, client_id, client_features, top_k)

        if true_policy in recommended_policies:
            correct_predictions += 1

        total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Model Top-{top_k} Accuracy: {accuracy:.4f}")
    return accuracy

model = tf.keras.models.load_model('XLMicro_v3.keras')

"""How well does the model generalize for new, unseen users?"""

evaluate_model(model, test, top_k=5)

"""How well does the model generalize for users it already knows?"""

evaluate_model(model, train, top_k=2)

"""# Workload Management System #"""



import random

class Agent:
    def __init__(self, id, effectiveness_rating, confident_policies):
        self.id = id
        self.effectiveness_rating = effectiveness_rating
        self.expertise = confident_policies  # List of policies agent is confident in
        self.clients = set()
        self.num_clients = 0
        self.is_available = True  # Starts available

    def assign_client(self, client_id):
        # Flag to check if the agent is overworked
        if not self.is_available:
            raise ValueError(f"Agent {self.id} has reached max capacity")
        else:
            self.clients.add(client_id)
            self.num_clients += 1
            if self.num_clients > 10:  # Max capacity for agent
                self.is_available = False

    def update_availability(self):
        # Dynamically update availability based on workload
        if self.num_clients <= 10:
            self.is_available = True
        else:
            self.is_available = False

    def unassign_client(self, client_id):
        self.clients.remove(client_id)
        self.num_clients -=1
        return

class Cluster:
    def __init__(self, label, product_col):
        self.label = label
        self.available_agents = 0
        self.agents = []
        self.product_col = product_col
        self.agent_tracker = {}

    def get_agents(self, data):
        agent_ids = data['agntnum'].unique()
        agent_effectiveness = data[f'pct_prod_{self.product_col}_cnvrt']  # Effectiveness rating
        agent_confident_policies = data['agent_product_expertise']  # List of policies each agent is confident in

        # Function to check confidence based on the product column
        def confident(policy_id, agent_confident_policies):
            if policy_id in agent_confident_policies:
                return 1
            else:
                return 0

        # Create agents and sort them based on effectiveness and confidence
        self.agents = sorted(
            [
                Agent(agent_id, rating, expertise)
                for agent_id, rating, expertise in zip(agent_ids, agent_effectiveness, agent_confident_policies)
            ],
            key=lambda x: (x.effectiveness_rating, confident(self.product_col, x.expertise)),
            reverse=True
        )

        for agent in self.agents:
            self.agent_tracker[agent.id] = agent

        self.available_agents = len(self.agents)

    def assign_next_available_agent(self, client_id):
        for agent in self.agents:
            if agent.is_available:
                agent.assign_client(client_id)
                return agent.id
        # No available agent in this cluster
        return 0

class WorkloadManager:
    def __init__(self, agent_df):
        self.clusters = {}
        self.total_agents = 0
        self.policy_to_cluster = {
            0: [5], 2: [7], 4: [4, 3],
            6: [9, 0], 7: [4], 8: [8, 1, 2],
            9: [5]
        }
        # set up client to (agent, cluster label) pairs
        self.client_assignments = {}
        self.unassigned_clients = []

        # Reverse mapping from cluster to policies (this needs to be a dictionary, not a set)
        self.cluster_to_policy = {}
        for policy_id, cluster_list in self.policy_to_cluster.items():
            for cluster_id in cluster_list:
                self.cluster_to_policy[cluster_id] = policy_id

        # Populate clusters based on agent_df
        for label in agent_df['cluster'].unique():
            # Get the relevant policy(s) for this cluster
            policy_for_cluster = self.cluster_to_policy.get(label)
            if policy_for_cluster is not None:
                cluster = Cluster(label, product_col=policy_for_cluster)  # Assign the appropriate policy's product column
                cluster_data = agent_df[agent_df['cluster'] == label]
                cluster.get_agents(cluster_data)
                self.clusters[label] = cluster

        self.total_agents = sum([cluster.available_agents for cluster in self.clusters.values()])

    def assign_client_to_agent(self, recommended_policies, client_id):
        if client_id in self.client_assignments.keys():
            print(f"Client with id {client_id} has already been assigned.")
            return

        # Attempt to assign a client to an available agent across recommended clusters
        policy_queue = recommended_policies[:]

        while policy_queue:
            policy_id = random.choice(policy_queue)
            recommended_clusters = self.policy_to_cluster.get(policy_id, [])

            for cluster_id in recommended_clusters:
                cluster = self.clusters.get(cluster_id, None)
                if cluster:
                    res = cluster.assign_next_available_agent(client_id)
                    if res != 0:
                        # Successfully assigned, record assignment
                        self.client_assignments[client_id] = [res, cluster.label]
                        print(f"Client {client_id} assigned to agent {res} in cluster {cluster.label}")
                        return

                else:
                    print(f"Cluster {cluster_id} does not exist for policy {policy_id}")

            # If all clusters for the current policy are full, remove the policy and reattempt with a new one
            policy_queue.remove(policy_id)

        # If all policies are exhausted and no assignment was made, raise an exception
        print(f"Could not assign client {client_id} to any agent for the recommended policies.")
        self.unassigned_clients.append(client_id)
        return


    def update_agents_availability(self):
        # Update agent availability after each assignment
        for cluster in self.clusters.values():
            for agent in cluster.agents:
                agent.update_availability()

    def unassign_client(self, client_id):
        assigned_agent, cluster_label = self.client_assignments[client_id][0], self.client_assignments[client_id][1]
        print(assigned_agent, cluster_label)
        cluster = self.clusters[cluster_label]
        cluster.agent_tracker[assigned_agent].unassign_client(client_id)
        print(f"Client {client_id} has been unassigned from {assigned_agent} in Cluster {cluster_label}")
        return

"""# Full Integration #"""

final_model = tf.keras.models.load_model('XLMicro_v3.keras')

TOP_K=3

def deploy():
  # Step 1: create global workload manager
  workload_manager = WorkloadManager(agent_df)
  for sample in test:
    # Step 2: get the recommended_policies
    client_id = sample["client_id"].numpy()
    true_policy = sample["policy_id"].numpy()
    client_features = sample["client_features"].numpy()
    recommended_policies, _ = recommend_policies(final_model, client_id, client_features, top_k=TOP_K)
    # Step 3: assign clients accordingly
    workload_manager.assign_client_to_agent(recommended_policies, client_id)

  workload_manager.update_agents_availability()
  print(f"Workload Manager with {workload_manager.total_agents} agents created.")
  return workload_manager

workload_manager = deploy()

# now we can see the features of the workload manager

workload_manager.client_assignments

workload_manager.clusters[1].agents[0].id