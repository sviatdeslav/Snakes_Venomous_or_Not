from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Загружаем модель
def load_model(model_path):
    model = SnakeVenomClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Переводим в режим оценки
    return model

# Преобразования для входного изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Функция для предсказания
def predict(image_path, model):
    # Загружаем и преобразуем изображение
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Добавляем размерность батча
    
    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    
    return predicted.item(), probabilities.squeeze().tolist()

# Загружаем модель (укажите правильный путь)
model = load_model('snake_venom_classifier.pth')

# Путь к вашему тестовому изображению
image_path = '/content/im.png'  # Замените на реальный путь

# Получаем предсказание
prediction, probs = predict(image_path, model)
class_names = ['Venomous', 'Non-Venomous']

# Выводим результат
print(f'\nPrediction: {class_names[prediction]}')
print(f'Confidence: {probs[prediction]*100:.2f}%')
print(f'\nDetailed probabilities:')
for name, prob in zip(class_names, probs):
    print(f'{name}: {prob*100:.2f}%')

# Показываем изображение
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.title(f'Predicted: {class_names[prediction]} ({probs[prediction]*100:.1f}%)')
plt.show()
