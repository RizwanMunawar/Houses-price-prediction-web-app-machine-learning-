import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'USA_Housing_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        aai = flask.request.form['aai']
        aah = flask.request.form['aah']
        aan = flask.request.form['aan']
        aanb = flask.request.form['aanb']
        ap = flask.request.form['ap']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[aai,aah,aan,aanb,ap]],
                                       columns=['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('index.html',
                                     original_input={'Avg. Area Income':aai,
                                                     'Avg. Area House Age':aah,
                                                     'Avg. Area Number of Rooms':aan,
			         								'Avg. Area Number of Bedrooms':aanb,
			         								'Area Population':ap	
				},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()