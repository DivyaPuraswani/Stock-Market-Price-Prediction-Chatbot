from flask import Flask, render_template, request

import chat as ch
app = Flask(__name__)
c = ch.chat()
c.model_load()
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/img")
def img():
    return render_template("image.html")

@app.route("/get")
def chatter():
    userText = request.args.get('msg')
    print(userText)
    m = str(c.chatter(userText.lower()))
    print ("app.py",m)
    return m

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
