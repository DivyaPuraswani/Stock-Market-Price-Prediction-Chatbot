from flask import Flask, render_template, request

import chat as ch
import global_var as gv
app = Flask(__name__)
c = ch.chat()
c.model_load()
image = gv.global_var()


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/img")
def img():
    return render_template('image.html', name = 'Graph Plot', url ='/static/graph'+image.get_img()+'.jpeg')

@app.route("/get")
def chatter():
    userText = request.args.get('msg')
    print(userText)
    m = str(c.chatter(userText.lower())).split('</a>')
    print ("app.py",m)
    if len(m) > 1:
        image.set_img( m[1])
    return m[0]

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
