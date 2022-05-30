
from flask import Flask, redirect ,render_template,request,url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from numpy import loadtxt
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# load model
model = load_model('rnn_model.h5')

app=Flask(__name__)

app.config['SQ;ALCHEMY_DATABASE_URL']='sqlite///test.db'
db=SQLAlchemy(app)

class Todo(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    content=db.Column(db.String(200),nullable=False)
    

    def __repr__(self):
        return '<Task %r>' %self.id

db.create_all()

@app.route('/', methods=['POST','GET'])
def index():
    if request.method=='POST':
        #result =request.form['firstname']
        tokenizer = Tokenizer()
        seed_text = request.form['firstname']
        max_sequence_len = 9
        next_words = 50
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted = np.argmax(model.predict(token_list), axis=1)
            # model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                break
            seed_text += " " + output_word

        line = seed_text.split()
        for index, item in enumerate(line):
            if (index + 1) % 5 == 0:
                print(item)
            else:
                print(item, end=" ")
        try:
            return render_template('index.html',result=line )
        except:
            return 'There was issue'


    
    else:
        return render_template('index.html')





if __name__ =="__main__":
    app.run(debug=True)
