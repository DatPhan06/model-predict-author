import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Đọc dữ liệu từ tệp CSV
train_df = pd.read_csv('./train/train.csv')

# Hiển thị năm dòng đầu tiên của DataFrame để kiểm tra
print(train_df.head())

# Giả sử 'train_df' là DataFrame chứa dữ liệu huấn luyện
X = train_df['text']
y = train_df['author']

# Chuyển đổi nhãn lơớp thành dạng số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)

# Tiền xử lý dữ liệu văn bản
vectorizer = TfidfVectorizer(max_features = 1100)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

print(X_train_tfidf.shape)
print(X_train_tfidf[100])



model = Sequential(
    [
        tf.keras.Input(shape = (X_train_tfidf.shape[1],)),
        Dense(units = 35, activation = "relu"),
        Dropout(0.3),
        Dense(units = 15, activation = "relu"),
        Dropout(0.5),
        Dense(units = 20, activation = "relu"),
        Dense(units = 3, activation='softmax'),

    ], name = "my_model"
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
)

history = model.fit(
    X_train_tfidf, y_train, epochs = 30, batch_size = 40, validation_data = (X_test_tfidf, y_test)
)

# Đánh giá mô hình
loss = model.evaluate(X_test_tfidf, y_test)
print("Test Loss:", loss)

# Dự đoán và tạo file CSV cho submission

sub_df = pd.read_csv('./test/test.csv')
X_sub = sub_df['text']
X_sub_tfidf = vectorizer.transform(X_sub).toarray()

predictions = model.predict(X_sub_tfidf)

submission_df = pd.DataFrame({'id':  sub_df['id'], 'EAP': predictions[:, 0], 'HPL': predictions[:, 1], 'MWS': predictions[:, 2]})
submission_df.to_csv('submission.csv', index = False)

submission_df.head()