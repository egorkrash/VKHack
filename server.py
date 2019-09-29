from flask import Flask, request, jsonify
from sentiment import sent_predict, sent_analyse_dates
from cluster import infer_cluster, get_top_clusters_overall, get_top_clusters_month, get_top_clusters_week, comments
import numpy as np
app = Flask(__name__)


@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    if request.method == 'POST':
        input_dict = request.get_json()
        text = input_dict['text']
        preds = sent_predict([text])

        #code
        negative_estimate = np.round(100 * preds[0][1], 2)
        positive_estimate = np.round(100 * preds[0][0], 2)
        res = {'positive_estimate': positive_estimate, 'negative_estimate' : negative_estimate}
        return jsonify(res)

@app.route('/topic_modelling', methods=['POST'])
def topic_modelling():
    if request.method == 'POST':
        input_dict = request.get_json()
        text = input_dict['text']

        #code
        cluster_name, random_messages = infer_cluster(text)

        res = {'cluster_name': str(cluster_name),
         'message_1' : random_messages[0],
         'message_2' : random_messages[1],
         'message_3' : random_messages[2]}

        return jsonify(res)


@app.route('/get_html_map', methods=['GET'])
def get_html_map():
    if request.method == 'GET':
        with open('lda.html', 'r') as f:
            html = f.read()

        res = {'html_text': html}
        return jsonify(res)



@app.route('/get_topic_map', methods=['POST'])
def get_topic_map():
    if request.method == 'POST':
        input_dict = request.get_json()
        time = input_dict['time']

        possible_time = {'week', 'month', 'all'}

        if time not in possible_time:
            return 'this option is not possible'
        #code
        if time == 'week':
             pred = get_top_clusters_week(comments)
        elif time == 'month':
             pred = get_top_clusters_month(comments)
        else:
             pred = get_top_clusters_overall(comments)

        keys, values = list(pred.keys()), list(pred.values())
        clusters_ratio = {'cluster_1_name': str(keys[0]),
               'cluster_2_name': str(keys[1]),
               'cluster_3_name': str(keys[2]),
               'cluster_1_ratio': values[0],
               'cluster_2_ratio': values[1],
               'cluster_3_ratio': values[2]
               }
        #clusters_ratio = {
        #       str(keys[0]): values[0],
        #       str(keys[1]): values[1],
        #       str(keys[2]): values[2]
        #       }
        return jsonify(clusters_ratio)

@app.route('/get_sentiment_map', methods=['POST'])
def get_sentiment_map():
    if request.method == 'POST':
        input_dict = request.get_json()
        time = input_dict['time']

        possible_time = {'week', 'month', 'all'}

        if time not in possible_time:
             return 'this option is not possible'
        #code
        mean_scores = sent_analyse_dates(time)
        negative_estimate = np.round(100 * mean_scores[1], 2)
        positive_estimate = np.round(100 * mean_scores[0], 2)
        res = {'positive_estimate': positive_estimate, 'negative_estimate': negative_estimate}
        return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
