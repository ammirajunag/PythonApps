# app.py
from flask import Flask

app = Flask(__name__)

num = 0

@app.route('/')
def hello_world():
    for i in range(5):
        num += 1
        print(num)
    return 'Hello, World!' + str(num)

if __name__ == '__main__':
    app.run(debug=True)
