import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from flask import Flask
from flask import jsonify

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

data_path = '../data/21B_tag_views_dataset.csv'
app = Flask("recommender_api")


def get_data(path: str):
    return pd.read_csv(path)

#################################
# User Based Collaborative Model
#################################


def _user_data_preparation(data: pd.DataFrame):
    tag_count_df = data.groupby(['user_id', 'tag_id']).agg({'tag_id': 'count'}).rename(columns={'tag_id': 'tag_count'})
    tag_count_df = tag_count_df.reset_index()
    user_products_df = tag_count_df.pivot(index='user_id', columns='tag_id', values='tag_count').fillna(0)
    return user_products_df


def _euclidean_distance_user_model(user_tags_df: pd.DataFrame):
    item_similarity = pairwise_distances(user_tags_df.values.T, metric='euclidean')
    user_predictions = user_tags_df.values.dot(item_similarity)/np.abs(item_similarity.sum(axis=1))
    user_predictions = 1 / (1 + user_predictions)
    return pd.DataFrame(user_predictions, columns=user_tags_df.columns, index=user_tags_df.index)


def generate_user_model(data: pd.DataFrame):
    user_products_df = _user_data_preparation(data)
    predictions_df = _euclidean_distance_user_model(user_products_df)
    return predictions_df


def get_user_predictions(data, user, predictions, n_preds, not_seen=True):
    """
    data: original user data
    user: str user id
    predictions: pd.DataFrame of user_id/tag_id with predictions 'score'
    n_preds: int number of predictions to return
    not_seen: bool flag to return user seen predictions or not
    """
    user_predictions = predictions.loc[user].sort_values(ascending=False).reset_index()
    user_products = data.loc[user]['tag_id']
    if not_seen:
        user_predictions = user_predictions[~user_predictions['tag_id'].isin(user_products)]

    return user_predictions[:n_preds]['tag_id'].values.tolist()


#################################
# Item Based Collaborative Model
#################################


def _tag_data_preparation(data: pd.DataFrame):
    tag_count_df = data.groupby(['user_id', 'tag_id']).agg({'tag_id': 'count'}).rename(columns={'tag_id': 'tag_count'})
    tag_count_df = tag_count_df.reset_index()
    tags_user_df = tag_count_df.pivot(index='tag_id', columns='user_id', values='tag_count').fillna(0)
    return tags_user_df


def _cosine_distance_model(tags_user_df: pd.DataFrame):
    item_distances = pairwise_distances(tags_user_df.values, metric='cosine')
    item_similarity = 1 - item_distances
    np.fill_diagonal(item_similarity, 0)
    return pd.DataFrame(item_similarity, columns=tags_user_df.index, index=tags_user_df.index)


def generate_tag_model(data: pd.DataFrame):
    tag_user_data = _tag_data_preparation(data)
    item_similarity = _cosine_distance_model(tag_user_data)
    return item_similarity


def get_tag_predictions(item_similarity_scores, tag, top_n):
    items_keep = item_similarity_scores.loc[tag][item_similarity_scores.loc[tag] > 0]
    return items_keep.sort_values(ascending=False)[:top_n].index.tolist()


# Load the data and prepare the user model
logger.info("Generating user predictions")
user_data = get_data(data_path)
user_data = user_data.set_index('user_id')
user_predictions = generate_user_model(user_data)

logger.info("Generating tag predictions")
tag_data = get_data(data_path)
tag_similarities = generate_tag_model(tag_data)
tag_data = tag_data.set_index('tag_id')

logger.info("Data and model generated successfully")
@app.route('/predict-user/<string:user_id>')
def get_user_recommended_tags(user_id):
    # Check that user exists:
    try:
        user_data.loc[user_id]
    except:
        return jsonify({"error": "The user does not exist"})
    user_predicted_tags = get_user_predictions(user_data, user_id, user_predictions, 10, not_seen=True)
    return jsonify({"predicted_tags": user_predicted_tags})


@app.route('/predict-tag/<string:tag_id>')
def get_tag_recommended_tags(tag_id):
    # Check that tag exists:
    try:
        tag_data.loc[tag_id]
    except:
        return jsonify({"error": "The tag does not exist"})
    tag_predicted_tags = get_tag_predictions(tag_similarities, tag_id, 10)

    return jsonify({"predicted_tags": tag_predicted_tags})


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=80)

