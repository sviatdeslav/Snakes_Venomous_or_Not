import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
import os
from tqdm import tqdm

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Путь к датасету
data_dir = "/content/drive/MyDrive/Snakes-Dataset-split"

# Определение трансформаций для данных
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224), # Случайно вырезает область из изображения и изменяет её размер до 224×224 пикселей.
    transforms.RandomHorizontalFlip(), # Случайно отражает изображение по горизонтали (слева направо) с вероятностью 50%
    transforms.RandomRotation(20), # : Случайно поворачивает изображение на угол от -20° до +20°
    transforms.ToTensor(), # Преобразование в тензор
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Нормализует тензор по каналам (RGB) с заданными средним и стандартным отклонением
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256), # Изменяет размер изображения так, чтобы меньшая сторона стала 256 пикселей
    transforms.CenterCrop(224), #  Вырезает центральную область изображения размером 224×224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка датасетов
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'train'),
    train_transforms
)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'val'),
    val_test_transforms
)

test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'test'),
    val_test_transforms
)

# Создание DataLoader
batch_size = 32

train_loader = DataLoader(
    train_dataset,      # объект датасета
    batch_size=batch_size,      # размер батча
    shuffle=True,       # перемешивать ли данные
    num_workers=2       # количество потоков для загрузки
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# Инициализация модели, функции потерь и оптимизатора
model = SnakeVenomClassifier().to(device)
criterion = nn.CrossEntropyLoss() # Кросс-энтропийная потеря
optimizer = optim.Adam(model.parameters(), lr=0.001) # Адам

# Функция для обучения
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time() # засекает время начала обучения
    best_model_wts = model.state_dict() # сохраняет веса модели
    best_acc = 0.0 # отслеживает лучшую точность на валидации
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Каждая эпоха имеет фазы обучения и валидации
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Режим обучения
                dataloader = train_loader
            else:
                model.eval()   # Режим оценки
                dataloader = val_loader
            running_loss = 0.0 # накапливает значение функции потерь
            running_corrects = 0 # считает количество верных предсказаний
            # Итерация по данным
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Обнуляем градиенты
                optimizer.zero_grad()
                # Прямой проход
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # определяет предсказанные классы
                    loss = criterion(outputs, labels)
                    # Обратный проход + оптимизация только в фазе обучения
                    if phase == 'train':
                        loss.backward() # вычисляет градиенты
                        optimizer.step() # обновляет веса модели
                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # Глубокое копирование модели, если это лучшая точность на валидации
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    # Загрузка лучших весов модели
    model.load_state_dict(best_model_wts)
    return model

# Обучение модели
model = train_model(model, criterion, optimizer, num_epochs=25)

# Тестирование модели
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

test_model(model, test_loader)

# Сохранение модели
torch.save(model.state_dict(), 'snake_venom_classifier.pth')
