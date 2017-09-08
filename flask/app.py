from flask import Flask
from flask import render_template, request

app = Flask(__name__)

@app.route('/')
@app.route('/<name>') # <name> positional argument
def index(name = None):
    name = request.args.get('name')
    return render_template('index.html', name=name, fixed='some string')

if __name__ == '__main__':
    app.run(port=4998, debug=True)
