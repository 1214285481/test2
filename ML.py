#!/usr/bin/env python
# coding: utf-8

# In[1]:


#一、训练模型
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import joblib
import tkinter as tk
from tkinter import filedialog
import pandas as pd


# In[2]:


# 2、使用Tkinter创建图形界面来选择文件
#2.1初始化Tkinter根窗口
root = tk.Tk()
#2.2隐藏Tkinter主窗口
root.withdraw()
#2.3打开文件对话框选择文件。askopenfilename是tkinter.filedialog模块中的一个函数，
#它会打开一个文件选择对话框，允许用户选择一个文件。filetypes参数是一个元组列表，
#用来指定对话框中可选择的文件类型。这里设置为[("Excel files", "*.xlsx")]，
#意味着对话框将仅显示扩展名为.xlsx的Excel文件。
file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
# 2.4读取Excel文件。使用pandas库的read_excel函数读取用户选择的Excel文件。
#file_path变量包含了文件的完整路径。这行代码将Excel文件加载到data的DataFrame对象中，
#DataFrame是pandas中用于存储表格数据的主要数据结构。这使得之后可以进行数据处理、分析和可视化。
df = pd.read_excel(file_path)


# In[3]:


#2.5定义特征。从data DataFrame中移除名为Group的列，将剩余的数据存储在新的DataFrame X中。
#在这里，Group列通常代表了数据的目标变量或标签，而X包含了作为模型输入的特征。
X = df.drop(columns=['class'])
#2.6定义标签。将data DataFrame中的Group列单独提取出来，赋值给变量y。
#在机器学习中，y通常用来存储目标变量，即模型需要预测的变量。
y = df['class']# 定义标签
# 2.7分割数据集为训练集和测试集。
#使用train_test_split函数将特征集X和标签集y随机划分为训练集和测试集。
#参数test_size=0.2表示20%的数据将被划为测试集，剩下的80%则作为训练集。
#random_state=42是一个种子值，用来确保每次划分时能够重现相同的结果，这对于科学研究和问题排查是很有帮助的。
#(X_train, y_train): 这部分数据用于训练机器学习模型，即模型将在这些数据上学习特征与标签之间的关系。
#(X_test, y_test): 这部分数据用于测试，它允许研究者评估模型在未见过的数据上的表现，从而测试其泛化能力。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


#训练RF分类器
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X_train, y_train)


# In[5]:


#评估模型
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
target_names=["Non-MVA","MVA"]
report=classification_report(y_test,y_pred,target_names=target_names)
print(f"Model Accuracy:{accuracy}")
print("Classification Report:")
print(report)          


# In[6]:


#保存模型
joblib.dump(clf,"RF_model.pkl")


# In[7]:


get_ipython().system('python train_model.py')


# In[61]:


#使用FastAPI构建ML API
#!pip install fastapi uvicorn


# In[102]:


from fastapi import FastAPI,Request,Form,HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


# In[103]:


app=FastAPI()


# In[105]:


#下载训练模型
model=joblib.load("RF_model.pkl")


# In[106]:


#target_names=["Non-MVA","MVA"]
from pydantic import BaseModel,Field
class dfInput(BaseModel):
    SBP:float    
    Killip_class:int=Field(...,ge=1,le=4,description="Killip分级: 1=I, 2=II, 3=III, 4=IV")
    History_of_drinking:int=Field(...,ge=0,le=1,description="饮酒史: 0=否, 1=是")
    CK_MB:float
    Serum_potassium_at_admission:float
    Serum_potassium_at_24h_after_PCI:float
    Monocytes:float
    NLR:float
    SII:float


# In[107]:


class dfPrediction (BaseModel):
    predicted_class:int
    predicted_class_name:str


# In[108]:


@app.post("/predict")
async def predict(input_data:dfInput):
    try:
        input_array=np.array([[
            input_data.SBP,
            input_data.Killip_class,
            input_data.History_of_drinking,
            input_data.CK_MB,
            input_data.Serum_potassium_at_admission,
            input_data.Serum_potassium_at_24h_after_PCI,
            input_data.Monocytes,
            input_data.NLR,
            input_data.SII]])
        prediction=model.predict(input_array)[0]
        return{"prediction":str(prediction)}
    except Exception as e:
        return{"error":str(e)}            


# In[110]:


if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    
    nest_asyncio.apply()
    uvicorn.run(app,host="127.0.0.1",port=8000)


# In[158]:


get_ipython().system('python app.py')


# In[159]:


#为Web应用程序构建UI
from typing import Literal
from fastapi import FastAPI, Request,Form,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np


# In[161]:


model=joblib.load("RF_model.pkl")


# In[162]:


app=FastAPI()


# In[163]:


templates=Jinja2Templates(directory="template")


# In[164]:


target_names=["Non-MVA","MVA"]


# In[153]:


#class dfInput (BaseModel):
    #SBP:float
    #Killip_class:int
    #History_of_drinking:int
    #CK_MB:float
    #Serum_potassium_at_admission:float
    #Serum_potassium_at_24h_after_PCI:float
    #Monocytes:float
    #NLR:float
    #SII:float


# In[165]:


class dfPrediction (BaseModel):
    predicted_class:str
    predicted_class_name:str


# In[166]:


@app.get("/",response_class=HTMLResponse)
async def read_root(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})


# In[167]:


@app.post("/predict",response_model=dfPrediction)
async def predict(
    request:Request,
    SBP:float=Form(...),
    Killip_class:int=Form(...,ge=1,le=4),
    History_of_drinking:int=Form(...,ge=0,le=1),
    CK_MB:float=Form(...),
    Serum_potassium_at_admission:float=Form(...),
    Serum_potassium_at_24h_after_PCI:float=Form(...),
    Monocytes:float=Form(...),
    NLR:float=Form(...),
    SII:float=Form(...),):
    try:
        input_data=pd.DataFrame([[SBP, Killip_class, History_of_drinking, CK_MB,
                                  Serum_potassium_at_admission,Serum_potassium_at_24h_after_PCI,
                                  Monocytes,NLR,SII]],columns=["SBP", "Killip_class", "History_of_drinking", "CK_MB",
                                                               "Serum_potassium_at_admission",
                                                               "Serum_potassium_at_24h_after_PCI",
                                                               "Monocytes","NLR","SII" ])
        predicted_class_name=model.predict(input_data)[0]
        predicted_class=0 if predicted_class_name=="Non-MVA" else 1
        
        return templates.TemplateResponse(
            "result.html",
            {
                "request":request,
                "predicted_class":predicted_class,
                "predicted_class_name":predicted_class_name,
                "SBP":SBP,
                "Killip_class":Killip_class,
                "History_of_drinking":History_of_drinking,
                "CK_MB":CK_MB,
                "Serum_potassium_at_admission":Serum_potassium_at_admission,
                "Serum_potassium_at_24h_after_PCI":Serum_potassium_at_24h_after_PCI,
                "Monocytes": Monocytes,
                "NLR":NLR,
                "SII":SII})
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))


# In[ ]:


if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app,host="127.0.0.1",port=8000)

