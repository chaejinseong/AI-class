import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 CSV 파일 읽기
file_path = "/Users/chaejinseong/aistart/WEEK0/iris.csv"  # 본인이 iris.csv를 저장한 경로를 입력
df = pd.read_csv(file_path)

# 데이터프레임 확인
df.head()
print(df.columns)

# 특징(X)과 타겟(y) 분리
X = df.drop(columns=['Name'])  # 'Name'이 정답(label) 컬럼일 가능성이 높음
y = df['Name']

# 문자열 라벨을 숫자로 변환
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree 모델 학습 및 평가
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# Random Forest 모델 학습 및 평가
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# SVM 모델 학습 및 평가
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Logistic Regression 모델 학습 및 평가
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
