from sklearn.model_selection import train_test_split
from google.colab import drive
import os
import shutil

drive.mount('/content/drive')
# Пути к данным
input_folder = "/content/drive/MyDrive/Snakes-Dataset"
output_folder = "/content/drive/MyDrive/Snakes-Dataset-split"
os.makedirs(output_folder, exist_ok=True)

# Собираем все файлы с метками классов
file_paths = []
labels = []

for class_name in ["Venomous", "Non-Venomous"]:
    class_dir = os.path.join(input_folder, class_name)
    for file in os.listdir(class_dir):
        file_paths.append(os.path.join(class_dir, file))
        labels.append(class_name)

# Разбиваем на train-val-test с сохранением пропорций
X_train, X_test, y_train, y_test = train_test_split(
    file_paths, labels, test_size=0.1, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.222, stratify=y_train, random_state=42  # 0.222 = 20% от оставшихся 90%
)

# Копируем файлы в целевые папки
def copy_files(file_paths, labels, target_dir):
    for file_path, label in zip(file_paths, labels):
        os.makedirs(os.path.join(target_dir, label), exist_ok=True)
        shutil.copy(file_path, os.path.join(target_dir, label, os.path.basename(file_path)))

copy_files(X_train, y_train, os.path.join(output_folder, "train"))
copy_files(X_val, y_val, os.path.join(output_folder, "val"))
copy_files(X_test, y_test, os.path.join(output_folder, "test"))

print("Стратифицированное разбиение завершено!")
