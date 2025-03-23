import pickle, bz2
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from app_logger import log
import warnings
warnings.filterwarnings("ignore")
import smtplib

# ------------ML---------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def predict_forest_fire(temp, wind, humidity, rain):
    sample_input = df.drop(columns=["area"]).mean().to_dict()
    sample_input.update({"temp": temp, "wind": wind, "RH": humidity, "rain": rain})
    input_df = pd.DataFrame([sample_input])
    input_scaled = scaler.transform(input_df)
    predicted_area = model.predict(input_scaled)[0]
    predicted_area = np.expm1(predicted_area)  # Reverse the log transformation
    
    # Threshold for classification
    if predicted_area > 5:
        # return "Forest is in Danger!"
        return 1
    else:
        # return "Forest is Safe!"
        return 0

def mail():
    HOST = "smtp.gmail.com"
    PORT = 587

    # Sender and Receiver Information
    FROM_EMAIL = "kunalkawate242@gmail.com"
    PASSWORD = "Your_Gmail_APP_Pass_Here"  # Use an app password instead of your actual password for security
    TO_EMAIL = "kunalkawate424@gmail.com"

    # Email Subject and Message
    subject = "Urgent Alert: Potential Forest Fire Detected!"
    message = """\
    We have detected a potential forest fire risk in your area based on recent environmental conditions.


    **Recommended Actions:**  
    - Avoid open flames and flammable materials in the affected area.  
    - Report any signs of fire to local authorities immediately.  
    - Follow emergency protocols if evacuation is required.  

    Your safety is our priority. Stay alert and take precautions!  

    **Best Regards,**  
    Forest Fire Alert System  
    """

    # Formatting the message correctly
    MESSAGE = f"Subject: {subject}\n\n{message}"

    # Sending the email
    try:
        smtp = smtplib.SMTP(HOST, PORT)
        smtp.ehlo()
        smtp.starttls()
        smtp.login(FROM_EMAIL, PASSWORD)
        smtp.sendmail(FROM_EMAIL, TO_EMAIL, MESSAGE.encode("utf-8"))
        smtp.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

app = Flask(__name__)
# Load the improved dataset
file_path = "improved_forest_fire_dataset_5000.csv"
df = pd.read_csv(file_path)

# Select features and target variable
X = df.drop(columns=["area"])
y = np.log1p(df["area"])  # Log transformation of target variable

# Normalize the features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting model with optimized parameters
model = GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred) * 100  # Convert to percentage
mae = mean_absolute_error(y_test, np.expm1(y_pred))
mse = mean_squared_error(y_test, np.expm1(y_pred))



# Route for homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')


@app.route('/predictC', methods=['POST', 'GET'])
def predictC():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Temperature=float(request.form['Temperature'])
            Wind_Speed =int(request.form['WS'])
            rain=float(request.form['rain'])
            HuM=float(request.form['HM'])
            prediction = predict_forest_fire(Temperature, Wind_Speed, HuM, rain)
            log.info('Prediction done for Classification model')
            if prediction == 0:
                text = 'Forest is Safe!'
            else:
                text = 'Forest is in Danger!'
                mail()

            return render_template('index.html', prediction_text1="{}".format(text))
        except Exception as e:
            log.error('Input error, check input', e)
        return render_template('index.html', prediction_text1="Check the Input again!!!")


if __name__ == "__main__":
    app.run(debug=True, port= 5000)

