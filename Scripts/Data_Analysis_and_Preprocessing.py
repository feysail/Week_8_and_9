import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers


def load_data(path):
    return pd.read_csv(path)

    
     
def clean_dupliate(path):
    df=pd.read_csv(path)
    return df.drop_duplicates(keep='first')

def univariate(df):
    numeric = df.select_dtypes(include='number').columns.tolist()
    fig, axes = plt.subplots(6, 5, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric):
        axes[i-1].hist(df[col], bins=30)  
        axes[i-1].set_title(col)           
        axes[i-1].set_ylabel('Frequency')   

    plt.tight_layout()                  
    plt.show()   
    
def boxplot(data):
    numeric_columns=data.select_dtypes(include=['number']).columns.to_list()
    num=(np.sqrt((len(numeric_columns))))
    if num is int:
        fig,axes=plt.subplots(num,num,figsize=(45,45))
        axes=axes.flatten()
        for i,col in enumerate(numeric_columns):
         print(axes[i-1].boxplot(data[col]))
         print(axes[i-1].set_title(col))
        plt.tight_layout()
        plt.show()
    
    else:
        num=int(num)
        fig,axes=plt.subplots(num+1,num,figsize=(45,45))
        axes=axes.flatten()
        for i,col in enumerate(numeric_columns):
         print(axes[i-1].boxplot(data[col]))
         print(axes[i-1].set_title(col))
        plt.tight_layout()
        plt.show()
          
    
    

def IQR_outliers(df):
    numeric_columns = df.select_dtypes(include='number').columns.to_list()
    new_df = df.copy()

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        new_df = new_df[(new_df[col] >= lower_limit) & (new_df[col] <= upper_limit)]

    return new_df


def standardize(data):
    numeric_columns = data.select_dtypes(include='number').columns.to_list()
    numeric_columns.remove('user_id')
    scaler = MinMaxScaler()
    data[numeric_columns]=scaler.fit_transform(data[numeric_columns])
    
    
def encode_data(data):
    df=pd.get_dummies(data,columns=['browser','sex','source'],drop_first=True)
    return df
                        

def logistic_regression(X_train, X_test, y_train, y_test):
   log_reg = LogisticRegression()
   log_reg.fit(X_train, y_train)
   y_pred = log_reg.predict(X_test)
   print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
   

def decision(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
    

def RandomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    
    
def Gradient_Boosting(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(max_iter=1000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print("MLP Accuracy:", accuracy_score(y_test, y_pred))
    

def evaluate_cnn(X_train, X_test, y_train, y_test, input_shape):
    cnn_model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') 
    ])
    
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
    return cnn_accuracy

if __name__ == "__main__":
    cnn_accuracy = evaluate_cnn(X_train, X_test, y_train, y_test, (width, height, channels))
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
