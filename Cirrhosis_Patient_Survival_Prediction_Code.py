#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore")

#veri setini yükleme
sns.set(style="white")
cirrhosis_df=pd.read_csv("cirrhosis.csv")
cirrhosis_df


# In[2]:


# Sayısal sütunları seçme
sayisal_sutunlar =cirrhosis_df.select_dtypes(include=np.number)


# In[3]:


# Korelasyon matrisini hesaplama
korelasyon_matrisi = sayisal_sutunlar.corr()

# Korelasyon matrisini ısı haritası olarak görselleştirme
sns.heatmap(korelasyon_matrisi, annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()


# In[4]:


#id stununu kaldırma
cirrhosis_df = cirrhosis_df.drop("ID", axis=1)
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)
# "cirrhosis_yeni.csv" adında yeni bir dosyaya kaydetme


# In[5]:


#N_Days stununu kaldırma
cirrhosis_df = cirrhosis_df.drop("N_Days", axis=1)
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[6]:


cirrhosis_df = cirrhosis_df.drop("Sex", axis=1)


# In[7]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cirrhosis_df[['Bilirubin', 'Cholesterol', 'Albumin']] = scaler.fit_transform(cirrhosis_df[['Bilirubin', 'Cholesterol', 'Albumin']])


# In[8]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cirrhosis_df[['Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']] = scaler.fit_transform(cirrhosis_df[['Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']])


# In[9]:


CL=cirrhosis_df[cirrhosis_df.Status=="CL"]
C=cirrhosis_df[cirrhosis_df.Status=="C"]
D=cirrhosis_df[cirrhosis_df.Status=="D"]
C.info()


# In[14]:


cirrhosis_df = cirrhosis_df.drop(cirrhosis_df.index[312:418], axis=0)
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[21]:


cirrhosis_df


# In[31]:


cirrhosis_df["Age"] = cirrhosis_df["Age"] / 365.25  # 365.25 yıl için ortalama gün sayısı


# In[32]:


from sklearn.preprocessing import LabelEncoder

# LabelEncoder'ı oluştur
encoder = LabelEncoder()

# "Edema" sütununu dönüştür
cirrhosis_df["Edema"] = encoder.fit_transform(cirrhosis_df["Edema"])

# Değişiklikleri kaydet
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[33]:


# "Ascites" sütununu dönüştür
cirrhosis_df["Ascites"] = encoder.fit_transform(cirrhosis_df["Ascites"])

# Değişiklikleri kaydet
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[34]:


# "Spiders" sütununu dönüştür
cirrhosis_df["Spiders"] = encoder.fit_transform(cirrhosis_df["Spiders"])

# Değişiklikleri kaydet
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[35]:


# "Drug" sütununu dönüştür
cirrhosis_df["Drug"] = encoder.fit_transform(cirrhosis_df["Drug"])

# Değişiklikleri kaydet
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[36]:


cirrhosis_df["Status"] = encoder.fit_transform(cirrhosis_df["Status"])

# Değişiklikleri kaydet
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[38]:


cirrhosis_df["Hepatomegaly"] = encoder.fit_transform(cirrhosis_df["Hepatomegaly"])

# Değişiklikleri kaydet
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[40]:


cirrhosis_df.head(5)


# In[41]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[42]:


from sklearn.preprocessing import LabelEncoder
# LabelEncoder'ı oluşturun
encoder = LabelEncoder()

# "Status" sütununu dönüştür
# "Status" sütunundaki değerleri "CL", "C" ve "D" sırasıyla 0, 1 ve 2 olarak kodla
cirrhosis_df["Status"] = encoder.fit_transform(cirrhosis_df["Status"])

# Değişiklikleri kaydedin
cirrhosis_df.to_csv("cirrhosis_yeni.csv", index=False)


# In[43]:


# Bağımsız değişkenleri (X) ve bağımlı değişkeni (y) ayırın
X = cirrhosis_df.drop("Status", axis=1) 
y = cirrhosis_df["Status"]

# Veri setini eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


# KNN sınıflandırıcıyı oluştur
knn = KNeighborsClassifier(n_neighbors=5)  # 5 komşu kullanarak

# Modeli eğitim verileriyle eğit
knn.fit(X_train, y_train)


# In[45]:


# Test verileri üzerinde tahminlerde bulun
y_pred = knn.predict(X_test)

# Modelin doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

# Sınıflandırma raporunu yazdır
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Karışıklık matrisini yazdır
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))


# In[47]:


# Histogram çizimi
plt.hist(cirrhosis_df["Status"], bins=3)
plt.xlabel("Sınıf")
plt.ylabel("Veri Sayısı")
plt.title("Sınıf Dağılımı")
plt.show()


# In[58]:


from imblearn.over_sampling import SMOTE
# SMOTE algoritmasını oluşturma
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Aşırı örnekleme uygulaması
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[59]:


# Aşırı örneklemeden sonra sınıf dağılımını kontrol etme
print(pd.DataFrame(y_resampled).value_counts())


# In[63]:


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Veri setini yükleyin
cirrhosis_df = pd.read_csv("cirrhosis_yeni.csv")

# Bağımsız değişkenleri (X) ve bağımlı değişkeni (y) ayırın
X = cirrhosis_df.drop("Status", axis=1)
y = cirrhosis_df["Status"]

# Veri setini eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri standartlaştırın (KNN için önemlidir)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE nesnesini oluşturun
smote = SMOTE(random_state=42, sampling_strategy='auto')

# Eğitim verilerini oversample edin
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# KNN sınıflandırıcıyı oluşturun
knn = KNeighborsClassifier(n_neighbors=5)

# Modeli oversample edilmiş verilerle eğitin
knn.fit(X_train_resampled, y_train_resampled)

# Modeli değerlendirin
y_pred = knn.predict(X_test)

# Performans ölçütlerini yazdırın
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Karar ağacı modelini oluşturma
model = DecisionTreeClassifier()

# Modelin eğitim verileri üzerinde eğitilmesi
model.fit(X_train, y_train)


# In[65]:


# Test verileri üzerinde tahminler yapma
y_pred = model.predict(X_test)

# Modelin doğruluğunu değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.2f}")

# Sınıflandırma raporunu yazdırma
print(classification_report(y_test, y_pred))


# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)


# Lojistik regresyon modelini oluşturma
model = LogisticRegression(random_state=42)


# Modelin eğitim verileri üzerinde eğitilmesi
model.fit(X_train_resampled, y_train_resampled)


# Test verileri üzerinde tahminler yapma
y_pred = model.predict(X_test)


# Modelin doğruluğunu değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.2f}")

# Sınıflandırma raporunu yazdırma
print(classification_report(y_test, y_pred))


# In[67]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)



# Basit Bayes sınıflayıcı oluşturma ve eğitme
model = GaussianNB()
model.fit(X_train, y_train)

# Tahmin oluşturma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)


# In[69]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Sınıflandırma için
# from sklearn.ensemble import RandomForestRegressor # Regresyon için
from sklearn.metrics import accuracy_score # Sınıflandırma için
# from sklearn.metrics import mean_squared_error # Regresyon için


smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Verileri eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rasgele Orman Modelini oluşturun
model = RandomForestClassifier(n_estimators=100, random_state=42) # Sınıflandırma için
# model = RandomForestRegressor(n_estimators=100, random_state=42) # Regresyon için

# Modeli eğitim verileri üzerinde eğitin
model.fit(X_train, y_train)

# Test verileri üzerinde tahminler yapın
y_pred = model.predict(X_test)

# Modelin performansını değerlendirin
accuracy = accuracy_score(y_test, y_pred) # Sınıflandırma için
# mse = mean_squared_error(y_test, y_pred) # Regresyon için

print(f"Doğruluk: {accuracy:.2f}") # Sınıflandırma için
# print(f"Ortalama Kare Hatası (MSE): {mse:.2f}") # Regresyon için


# In[ ]:




