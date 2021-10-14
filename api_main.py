import api_rest

app = api_rest.Sanic.get_app('api-rest')
app.run(host='0.0.0.0', port=8080, debug=True, workers=4)