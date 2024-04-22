#Grigorios Pachis
#math5785
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import linalg


trainset=open(r"C:\Users\gregd\Desktop\Machine Learning\car_train.txt", "r")
testset=open(r"C:\Users\gregd\Desktop\Machine Learning\car_test.txt", "r")
lines1 = trainset.readlines()  # Read all lines into a list
'''
HPtrain = [line.split()[0] for line in lines1 if line.strip()]
Weighttrain = [line.split()[1] for line in lines1 if line.strip()]
Cylintrain=[line.split()[2] for line in lines1 if line.strip()]
ccmtrain=[line.split()[3] for line in lines1 if line.strip()]
kmtrain=[line.split()[4] for line in lines1 if line.strip()]
'''
#Arxika skeftika na parv ta dedomena ws listes columns into lists
lines2 = testset.readlines()
HPtest = [line.split()[0] for line in lines2 if line.strip()]
Weighttest = [line.split()[1] for line in lines2 if line.strip()]
Cylintest=[line.split()[2] for line in lines2 if line.strip()]
ccmtest=[line.split()[3] for line in lines2 if line.strip()]
kmtest=[line.split()[4] for line in lines2 if line.strip()]

#epita skeftika oti metatrepontas ta se ena data frame tha einai pio eykolo na ta kalesw 
#aplos xreiazotan kapoia mikrh epeksergasia, 
#H prwth grammi eprepe na figei giati htan strings "# hp Weight .." kai meta na petatrapoyn ta str numbers se numerical
trainDF=pd.DataFrame({
   'Hp': [line.split()[0] for line in lines1 if line.strip()],
   'Weight' :  [line.split()[1] for line in lines1 if line.strip()],
   'Cylinders':[line.split()[2] for line in lines1 if line.strip()],
   'ccm':[line.split()[3] for line in lines1 if line.strip()],
   'km/l':[line.split()[4] for line in lines1 if line.strip()]
})
trainDF1=trainDF.drop(0)

testDF=pd.DataFrame({
   'Hp': [line.split()[0] for line in lines2 if line.strip()],
   'Weight' :  [line.split()[1] for line in lines2 if line.strip()],
   'Cylinders':[line.split()[2] for line in lines2 if line.strip()],
   'ccm':[line.split()[3] for line in lines2 if line.strip()],
   'km/l':[line.split()[4] for line in lines2 if line.strip()]
})
testDF1=testDF.drop(0)

for col in trainDF1.columns:
    trainDF1[col] = pd.to_numeric(trainDF1[col], errors='coerce')

for col in testDF1.columns:
    testDF1[col] = pd.to_numeric(testDF1[col], errors='coerce')

#H akoloythi synartisi kanei normalize to Dataframe , xwris ayto h diafora megethous twn features stelnei to loss se terastia megethi kai den ginetai na brethei lisi
def maximum_absolute_scaling(df):    
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled
    

trainDF1 = maximum_absolute_scaling(trainDF1)

testDF1 = maximum_absolute_scaling(testDF1)
#edw exw teleiwsei me ton sximatismo kai tin epeksergasia twn dataframes
#akoloythoyn se sxolia kapoies prakseis poy ekana kathos meletoysa ta dedomena wste na exw mia eikona toy pws sxetizontai
'''
print(trainDF1)
def AvgHP(column):
    n=1
    sumHp=0
    while n <= (len(column)-1):
        sumHp=sumHp+float(column[n])
        n+=1
    AVGhp=sumHp/len(column)
    return AVGhp
  
#print(AvgHP(kmtest))



#sns.lmplot(
 #   x='km/l', 
  #  y='Hp',  
   # data=trainDF1 ,     
    #hue='Cylinders'
#)

#plt.ylabel('Horsepower (Hp)')
#plt.xlabel('km/l')
#plt.title('km/l vs Horsepower Distribution (Regression)')
#plt.grid(True)  

#plt.show()
'''
X = trainDF1.drop('km/l', axis='columns').values  # Apo to train dataframe xwrizw ta features 

y=trainDF1['km/l'] # kai edw orizw to target variable
w = np.random.rand(X.shape[1]) #Dimioyrgw ta arxika bari poy antistixoyn sta features
b = 0 
m=len(y)

#orizw to loss function
def mse(y_true, y_pred):
    return (1/(2*m))*np.sum(np.square(y_true - y_pred))

learning_rate = 0.05 # edw piramatistika me arketa learning rates , to 0.05 moy fanike ikanopoihtiko
epochs = 50 # trexontas gia arxika 1000 epochs blepoyme oti to loss den meionetai poly meta ta prwta 50
loss_history = [] # ithela na apothikeysw to loss moy se mia lista wste na blepw ton rithmo poy metabaletai alla sto print moy bgazei diaforetikh klimaka apo aythn poy perimenw  
wlib=[w]

# Training the model
for epoch in range(epochs):
  y_pred = np.dot(X, w) + b
  m=len(y)
  loss = mse(y, y_pred)
  #print('w:',w,'-',learning_rate *(1/m)*(X.T.dot(y_pred - y))) ayto einai ena print poy me bohthise na dw pws metabalontai ta weights
  
  w =w- learning_rate *(1/m)*(X.T.dot(y_pred - y))
  #print(w)
  wlib.append(w)
   
  #b =b- learning_rate * np.mean(y_pred - y) # to b tha parameinei 0 logo zhtoymenoy ths askhshs , parolauta , einai emfanes oti tha htan beltisto na prosarmozetai analoga

  #loss_history=np.append(epoch,loss)
  plt.scatter(epoch,loss) # anti lipon gia to plot toy loss history apofasisa na to kanw me to plt.scatter gia kathe epoch,loss 
  if loss<1e-4 or linalg.norm(wlib[-1]-wlib[-2])<1e-4 :
      #print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")
      break
  
  #print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")

print("Dianisma th: ",w,"\n Learning rate: ",learning_rate,"\n epanalipseis: ",epoch," , ε=0.0001 , δ=0.0001 \n")



#print(wlib)
#plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function Over Training')
plt.grid(True)
plt.show()

# Now 'w' and 'b' contain the trained weights and bias



X1 = testDF1.drop('km/l', axis='columns').values  # Apo to test dataframe xwrizw ta features 
y1=testDF1['km/l'] # kai edw orizw to target variable

def Eth(b,a):
    return np.linalg.norm(b-a)
from sklearn.metrics import mean_squared_error
predicted=np.dot(X1,w)

print("Second Norm :",Eth(predicted,y1))
#print(mean_squared_error(predicted,y1))

#---------------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

# Assuming you already have your data in X and Y (features and target label)

# Create a linear regression model
model = LinearRegression()

# Train the model on your data
model.fit(X, y)

# Make predictions using the trained model
y_predicted = model.predict(X)

# (Optional) Evaluate the model performance using metrics like R-squared or mean squared error
from sklearn.metrics import mean_squared_error

# Assuming you already have your trained model (`model`) and predicted values (`y_predicted`)

# Calculate the mean squared error
mse = mean_squared_error(y, y_predicted)

# Print the mean squared error
print("Mean Squared Error:", mse)

