import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from flask import Flask
from flask import jsonify

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

data_path = 'data/21B_tag_views_dataset.csv'
app = Flask("recommender_api")


def get_data(path: str):
    return pd.read_csv(path)


def _data_preparation(data):
    tag_count_df = data.groupby(['user_id', 'tag_id']).agg({'tag_id': 'count'}).rename(columns={'tag_id': 'tag_count'})
    tag_count_df = tag_count_df.reset_index()
    user_products_df = tag_count_df.pivot(index='user_id', columns='tag_id', values='tag_count').fillna(0)
    return user_products_df


def _euclidean_distance_model(user_tags_df: pd.DataFrame):
    item_similarity = pairwise_distances(user_tags_df.values.T, metric='euclidean')
    user_predictions = user_tags_df.values.dot(item_similarity)/np.abs(item_similarity.sum(axis=1))
    user_predictions = 1 / (1 + user_predictions)
    return pd.DataFrame(user_predictions, columns=user_tags_df.columns, index=user_tags_df.index)


def generate_model(data):
    user_products_df = _data_preparation(data)
    predictions_df = _euclidean_distance_model(user_products_df)
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


# Load the data and prepare the model
logger.info("Loading user data and generating predictions")
user_data = get_data(data_path)
user_data = user_data.set_index('user_id')
user_predictions = generate_model(user_data)
logger.info("Data and model generated successfully")


@app.route('/recommend/<string:user_id>')
def get_recommended_tags(user_id):
    # Check that user exists:
    try:
        user_data.loc[user_id]
    except:
        return jsonify({"error": "The user does not exist"})
    user_predicted_tags = get_user_predictions(user_data, user_id, user_predictions, 10, not_seen=True)
    return jsonify({"user_predicted_tags": user_predicted_tags})


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5151)

