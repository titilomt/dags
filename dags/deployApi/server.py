from flask import Flask, request, jsonify, Response
import pickle

from validations import validate_input


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        validate_input(data)
        prediction = model.predict([[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])

        output = str(prediction[0])
        return jsonify({"Predict": output})
    except Exception as err:
        return Response(str(err), status=404, mimetype='application/json')


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
