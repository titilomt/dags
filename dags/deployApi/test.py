import unittest
from flask import request, jsonify

from server import app


class TestHome(unittest.TestCase):

    def test_get(self):
        test_app = app.test_client()
        response = test_app.get('/')
        self.assertEqual(200, response.status_code)

    def test_predict_should_fail_input_error(self):
        test_app = app.test_client()
        response = test_app.post('/api/predict', json={
            "bla": "bla bla"
        })

        self.assertEqual(404, response.status_code)

    def test_predict_should_retrive_predict_two(self):
        with app.test_client() as c:
            rv = c.post('/api/predict', json={
                "sepal_length": 3.2,
                "sepal_width": 2.2,
                "petal_length": 4.3,
                "petal_width": 3.4
            })
            output = rv.get_json()

            assert str(output['Predict']) == "2"


if __name__ == '__main__':
    unittest.main()
