from flask import Flask,render_template,request
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index2.html')
@app.route('/',methods=['POST'])
def getvalue():
    Genderr=request.form['Gender']
    Height=request.form['Height']
    Weight=request.form['Weight']
    Height=float(Height) /100
    Weight=int(Weight)
    if Genderr=='Female':
        Gender=0
    else :
         Gender=1
    Gender=float(Gender)
    from collections import OrderedDict
    import pandas as pd
    import joblib
    classifer = joblib.load("model.pkl")
    new_data=OrderedDict([('Gender',Gender),('Height',Height),('Weight',Weight)])
    new_data=pd.Series(new_data)

    from sklearn.preprocessing import RobustScaler
    new_data1=new_data.values.reshape(1,-1)
    y_predict=classifer.predict(new_data1)
    def give_status(x):
        if x== 0:
            return 'Extremely Weak'
        elif x== 1:
            return 'Weak'
        elif x== 2:
            return 'Normal'
        elif x== 3:
            return 'Overweight'
        elif x== 4:
            return 'Obesity'
        elif x== 5:
            return 'Extreme Obesity'
    status=give_status(y_predict)
    return render_template('pass.html',status=status,gender=Genderr,weight=Weight,height=Height)