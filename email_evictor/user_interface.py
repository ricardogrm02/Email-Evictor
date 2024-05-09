from flask import Flask, render_template, request
from spam_classfier import runModel, emailPrediction
app = Flask(__name__,"/static")   # Flask constructor 
  
# A decorator used to tell the application 
# which URL is associated function 
@app.route('/', methods=['GET', 'POST'])       
def hello(): 
    if request.method == 'POST':
        email_content = request.form.get("email")
        print(email_content)
        output = emailPrediction(email_content)
        return render_template("index.html", accuracy=output[0], training_time=output[1], prediction_result = output[2], user_email_input = email_content)
    else:
        # Handle GET request (e.g., initial page load)
        return render_template("index.html", accuracy=None, training_time=None, prediction_result = None, user_email_input = "No input history")
if __name__=='__main__': 
   app.run() 