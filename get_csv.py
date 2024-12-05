#write a python script that gets the /data POST request and prints the data

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def get_data():
    # Get the JSON data from the POST request
    data = str(request.get_data())
    client_id = request.args.get('client_id')
    
    # Print the data
    print("ClientID", client_id, data)

    # save data to csv file with client_id as folder name inside "experiment_data" folder
    with open(f"experiment_data/{client_id}/data.csv", "w") as file:
        file.write(data)
    
    # Return a response
    return jsonify({"message": "Data received", "data": data}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)