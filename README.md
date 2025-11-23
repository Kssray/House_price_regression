# House Price Regression  
Created by: **Kübra Yılmaz – Computer Engineering Student**

---

## 🇹🇷 Proje Açıklaması (Turkish)

Bu proje, **California Housing** veri setini kullanarak ortalama ev fiyatını tahmin eden bir makine öğrenimi regresyon modelidir.  
Model, aşağıdaki özellikleri kullanarak ev fiyatını (MedHouseVal) tahmin eder:

- Medyan gelir (MedInc)  
- Ev yaşı (HouseAge)  
- Ortalama oda sayısı (AveRooms)  
- Ortalama yatak odası sayısı (AveBedrms)  
- Nüfus (Population)  
- Ortalama hane genişliği (AveOccup)  
- Koordinatlar (Latitude, Longitude)

Proje; veri ölçeklendirme, eğitim-test ayrımı ve Linear Regression algoritmasını içermektedir.

---

## 🇬🇧 Project Description (English)

This project is a regression model built using the **California Housing dataset**.  
The model predicts median house value (MedHouseVal) based on the following features:

- Median income  
- Housing age  
- Average number of rooms  
- Average number of bedrooms  
- Population  
- Average occupancy  
- Geographic coordinates  

The workflow includes feature scaling, train-test splitting, and training a Linear Regression model.

---

## 📂 File Structure

```text
house-price-regression/
│── main.py
│── requirements.txt
└── README.md
