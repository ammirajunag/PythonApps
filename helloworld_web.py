# app.py
from flask import Flask

app = Flask(__name__)

num = 0

@app.route('/')
for i in range(5):
    num += 1
    print(num)
def hello_world():
    return 'Hello, World!' + str(num)

if __name__ == '__main__':
    app.run(debug=True)
