from flask import Flask, render_template_string

app = Flask(__name__)

# Initialize counter
counter = 0

@app.route('/')
def hello_world():
    global counter
    counter += 1
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hello World Counter</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
            }
            .container {
                font-size: 24px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <p>Hello World!</p>
            <p>Page visits: {{ counter }}</p>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html, counter=counter)

if __name__ == '__main__':
    app.run(debug=True)
