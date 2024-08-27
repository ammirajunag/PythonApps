from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/add', methods=['POST'])
def add_numbers():
    try:
        # Extract JSON data from the request
        data = request.get_json()
        # Extract numbers from the JSON data
        num1 = data.get('num1')
        num2 = data.get('num2')

        # Validate the input
        if not isinstance(num1, (int, float)) or not isinstance(num2, (int, float)):
            return jsonify({"error": "Invalid input, both num1 and num2 should be numbers"}), 400

        # Calculate the sum
        result = num1 + num2

        # Return the result as JSON
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
