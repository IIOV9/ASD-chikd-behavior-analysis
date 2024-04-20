import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the intents data from intents.json file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/community')
def community():
    return render_template('community.html')

@app.route('/story2')
def story2():
    return render_template('story2.html')

@app.route('/scientific_articles')
def scientific_articles():
    return render_template('scientific_articles.html')

@app.route('/MindBot/')
def mind_bot():
    return render_template('MindBot.html', intents=intents)

@app.route('/api/intents')
def get_intents():
    return jsonify(intents)
    
    



if __name__ == '__main__':
    app.run(debug=True)
