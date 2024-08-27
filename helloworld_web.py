# app.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    num = 0
    for i in range(5):
        num += 1
        return 'Hello, World!' + str(num)
        print(num)
    

if __name__ == '__main__':
    app.run(debug=True)
