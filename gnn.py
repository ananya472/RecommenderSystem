import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import faiss  # Import FAISS
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity

class Neo4jDataExtractor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def extract_views_data(self):
        with self.driver.session() as session:
            views_query = "MATCH (u:User)-[:VIEWED]->(s:Stream) RETURN u.userId AS userId, s.streamID AS streamID"
            views_data = session.run(views_query)
            views_list = [{"userId": record["userId"], "streamID": record["streamID"]} for record in views_data]
            views_df = pd.DataFrame(views_list)

            if views_df.empty:
                print("No views data found, please check your database for the VIEWED relationship.")
                return None
            
            return views_df

    def extract_stream_metadata(self):
        with self.driver.session() as session:
            metadata_query = """
            MATCH (s:Stream)-[:BELONGS_TO]->(c:Category)
            RETURN s.streamID AS streamID, c.name AS category
            """
            metadata_data = session.run(metadata_query)
            metadata_list = [{"streamID": record["streamID"], "category": record["category"]} for record in metadata_data]
            return pd.DataFrame(metadata_list)

    def extract_user_interests(self):
        with self.driver.session() as session:
            interests_query = """
            MATCH (u:User)-[:INTERESTED_IN]->(c:Category)
            RETURN u.userId AS userId, c.name AS interest
            """
            interests_data = session.run(interests_query)
            interests_list = [{"userId": record["userId"], "interest": record["interest"]} for record in interests_data]
            return pd.DataFrame(interests_list)

class GNNRecommender(torch.nn.Module):
    def __init__(self, num_features):
        super(GNNRecommender, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class RecommenderSystem:
    def __init__(self, views_df, metadata_df, interests_df):
        self.views_df = views_df
        self.metadata_df = metadata_df
        self.interests_df = interests_df
        self.gnn_model = None
        self.index = None  # For FAISS index
        self.user_matrix = self.create_user_stream_matrix()

    def create_graph_data(self):
        # Create a user-item interaction graph
        num_users = len(self.views_df['userId'].unique())
        num_streams = len(self.views_df['streamID'].unique())

        user_ids = self.views_df['userId'].astype('category').cat.codes
        stream_ids = self.views_df['streamID'].astype('category').cat.codes

        edge_index = torch.tensor(np.vstack((user_ids, stream_ids)), dtype=torch.long)

        x = torch.eye(num_users + num_streams)  # One-hot encoding for nodes

        return Data(x=x, edge_index=edge_index)

    def create_user_stream_matrix(self):
        # Create a user-stream interaction matrix
        return self.views_df.pivot(index='userId', columns='streamID', values='streamID').fillna(0)

    def build_faiss_index(self):
        user_matrix = self.user_matrix.values.astype('float32')
        faiss.normalize_L2(user_matrix)  # Normalize the user matrix
        self.index = faiss.IndexFlatL2(user_matrix.shape[1])  # Create a FAISS index for L2 distance
        self.index.add(user_matrix)  # Add user matrix to the FAISS index

    def train_gnn(self, data):
        self.gnn_model = GNNRecommender(num_features=data.x.size(1))
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        self.gnn_model.train()

        for epoch in range(100):  # Adjust epochs as needed
            optimizer.zero_grad()
            out = self.gnn_model(data)
            # Here, you can define a suitable loss function; for simplicity, we'll just use MSE
            loss = F.mse_loss(out, out)  # Dummy loss for illustration
            loss.backward()
            optimizer.step()

    def generate_embeddings(self):
        self.gnn_model.eval()
        with torch.no_grad():
            data = self.create_graph_data()
            embeddings = self.gnn_model(data)
            return embeddings

    def collaborative_filtering_recommendations(self, user_id):
        # Ensure the FAISS index is built before making recommendations
        if self.index is None:
            self.build_faiss_index()

        user_index = int(user_id) - 1  # Adjust for zero-based index
        user_vector = self.user_matrix.iloc[user_index].values.astype('float32').reshape(1, -1)
        faiss.normalize_L2(user_vector)  # Normalize the user vector

        # Search for the 5 nearest neighbors
        distances, indices = self.index.search(user_vector, 5)  # Find top 5 similar users

        recommended_streams = set()
        for similar_user in indices[0]:
            if similar_user != user_index:  # Exclude the user themselves
                recommended_streams.update(self.user_matrix.columns[self.user_matrix.iloc[similar_user].values > 0])

        return list(recommended_streams)

    def content_based_recommendations(self, user_id):
        user_streams = self.views_df[self.views_df['userId'] == user_id]['streamID'].tolist()
        user_categories = self.metadata_df[self.metadata_df['streamID'].isin(user_streams)]
        
        if user_categories.empty:
            print(f"No categories found for streams viewed by user {user_id}.")
            return []

    # Create a category matrix
        category_matrix = pd.get_dummies(self.metadata_df['category'])
        if category_matrix.empty:
            print("No category data available for content-based recommendations.")
            return []

    # Build FAISS index for the category matrix
        category_matrix_values = category_matrix.values.astype('float32').copy()  # Ensure C-contiguous
        faiss.normalize_L2(category_matrix_values)  # Normalize the category matrix
        category_index = faiss.IndexFlatL2(category_matrix_values.shape[1])  # Create a FAISS index
        category_index.add(category_matrix_values)  # Add category matrix to the FAISS index

    # Get the indices of the streams viewed by the user
        stream_indices = self.metadata_df[self.metadata_df['streamID'].isin(user_streams)].index.tolist()

    # Get the category embeddings for the viewed streams
        viewed_categories = category_matrix.iloc[stream_indices].values.astype('float32').copy()  # Ensure C-contiguous
        faiss.normalize_L2(viewed_categories)  # Normalize the viewed categories

    # Search for similar streams using FAISS
        distances, indices = category_index.search(viewed_categories, 5)  # Find top 5 similar streams

        recommended_streams = set()
        for idx_list in indices:
            for idx in idx_list:
                recommended_streams.add(self.metadata_df.iloc[idx]['streamID'])

        return list(recommended_streams)
    
    def interest_based_recommendations(self, user_id):
        user_interests = self.interests_df[self.interests_df['userId'] == user_id]['interest'].tolist()
    
        if not user_interests:
            print(f"No interests found for user {user_id}.")
            return []

    # Find streams that belong to the user's interests
        recommended_streams = self.metadata_df[self.metadata_df['category'].isin(user_interests)]['streamID'].tolist()
    
        return recommended_streams


    def hybrid_recommendations(self, user_id):
        collaborative_recommendations = self.collaborative_filtering_recommendations(user_id)
        content_based_recommendations = self.content_based_recommendations(user_id)
        # Assuming there's an interest_based_recommendations function defined
        interest_based_recommendations = self.interest_based_recommendations(user_id)

        combined_recommendations = set(collaborative_recommendations) | set(content_based_recommendations) | set(interest_based_recommendations)
        return list(combined_recommendations)

# Usage example
if __name__ == "__main__":
    uri = "bolt://localhost:7687"  # Adjust if your Neo4j server is running elsewhere
    user = "neo4j"                  # Your Neo4j username
    password = "12345678"           # Your Neo4j password

    extractor = Neo4jDataExtractor(uri, user, password)
    
    try:
        views_df = extractor.extract_views_data()
        metadata_df = extractor.extract_stream_metadata()
        interests_df = extractor.extract_user_interests()
        
        if views_df is not None and not metadata_df.empty and not interests_df.empty:
            recommender = RecommenderSystem(views_df, metadata_df, interests_df)
            gnn_data = recommender.create_graph_data()
            recommender.train_gnn(gnn_data)

            user_id = 2  # Change this to the user ID you want recommendations for
            recommendations = recommender.hybrid_recommendations(user_id)
            print(f"Recommended streams for user {user_id}: {recommendations}")
    finally:
        extractor.close()