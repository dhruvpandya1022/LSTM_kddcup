import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load dataset
# data = pd.read_csv('kddcup.data_10_percent_corrected')

kddcup = datasets.fetch_kddcup99()
data = pd.DataFrame(data=kddcup.data, columns=kddcup.feature_names)

# Define categorical and numerical features

cat_features = ['protocol_type', 'service', 'flag']
num_features = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 'num_compromised', 'num_root',
                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

# Encode categorical features using label encoding

le = LabelEncoder()
for col in cat_features:
    data[col] = le.fit_transform(data[col])

class_names = ['back','land','neptune','pod','smurf', 'teardrop','ftp_write','guess_passwd','imap','multihop','phf',
               'spy','warezclient','warezmaster','buffer-overflow','loadmodule','perl','rootkit', 'ipsweep','nmap',
               'portsweep','satan']

# One-hot encode categorical features

data = pd.get_dummies(data, columns=cat_features)
y_data = pd.get_dummies(kddcup.target, columns=class_names)

# Concatenate categorical and numerical features
if 'label' in data.columns:
    X = pd.concat([data.drop(['label'], axis=1), data['label']], axis=1)
else:
    X = data.copy()
y = y_data

# Split data into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data to a 3D tensor

X_train = np.reshape(X_train.to_numpy(), (X_train.shape[0], X_train.shape[1]))
X_val = np.reshape(X_val.to_numpy(), (X_val.shape[0], X_val.shape[1]))

# Build LSTM model
# model = Sequential()
# model.add(LSTM(64, input_shape=(1, X_train.shape[0]), activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# scaler = MinMaxScaler()

X_train = np.array(X_train).astype('float32')
y_train = np.array(y_train).astype('float32')
X_val = np.array(X_val).astype('float32')
y_val = np.array(y_val).astype('float32')

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

# X_train = np.nan_to_num(X_train)
# y_train = np.nan_to_num(y_train)
# X_val = np.nan_to_num(X_val)
# y_val = np.nan_to_num(y_val)

# X_train_norm = scaler.fit_transform(X_train)
# X_test_norm = scaler.transform(X_val)
# Y_train_norm = scaler.fit_transform(y_train)
# Y_test_norm = scaler.transform(y_val)

# Train model

print(X_train.shape)
print(y_train.shape)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

history_df = pd.DataFrame(history.history)

# Plot training and validation loss over epochs

plt.plot(history.history['loss'])
plt.grid()
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.grid()
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Plot heatmaps

cm = confusion_matrix(y_val_classes, y_pred_classes)
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, fmt='2.2f')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()