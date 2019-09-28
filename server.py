from flask import Flask, request, jsonify
from sentiment import sent_predict, sent_analyse_dates
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
        cluster_name = 'переводы за еду'
        random_messages = ['перевожу за шавуху', 'спасибо, шавуха была вкусная', 'офигенно поели']

        res = {'cluster_name': cluster_name,
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
        clusters_ratio = {'cluster_1_name': 'переводы за еду',
               'cluster_2_name': 'образование',
               'cluster_3_name': 'образование',
               'cluster_1_ratio': 23,
               'cluster_2_ratio': 19,
               'cluster_3_ratio': 17
               }
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
