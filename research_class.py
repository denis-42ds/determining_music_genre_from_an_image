# импорт модулей
import os
import torch
import faiss
import random
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, log_loss

# установка констант
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

class DatasetExplorer:
    def __init__(self, DATA_PATH=None, ASSETS_DIR=None):
        self.DATA_PATH = DATA_PATH
        self.ASSETS_DIR = ASSETS_DIR

    def explore_dataset(self):
        data = []
        genres = os.listdir(self.DATA_PATH)

        for genre in genres:
            genre_path = os.path.join(self.DATA_PATH, genre)
            album_images = os.listdir(genre_path)
            
            for image_name in album_images:
                data.append({'image_name': image_name, 'genre_name': genre})

        dataset = pd.DataFrame(data)

        print(f"Количество уникальных жанров: {dataset['genre_name'].nunique()}")
        print()
        print('Вывод общей информации:')
        dataset.info()
        print()
        print("Первые пять строк датафрейма:")
        display(dataset.head())

        plt.figure(figsize=(14, 6))
        df_agg = dataset.groupby('genre_name').agg({'image_name': 'count'}).reset_index()
        df_agg.rename(columns={'image_name': 'count_image_names'}, inplace=True)
        sns.barplot(data=df_agg,
                    x='genre_name',
                    y='count_image_names',
                    hue='genre_name',
                    palette=sns.color_palette("husl", len(df_agg)))

        plt.title(f"images total distribution")

        # сохранение графика в файл
        # plt.savefig(os.path.join(self.ASSETS_DIR, 'images total distribution.png'))
        plt.show()

        fig, axs = plt.subplots(3, 5, figsize=(14, 10))
        fig.suptitle('Examples of cover images', fontsize=20)
        for i, (index, row) in enumerate(dataset.sample(n=15).iterrows()):
            image_path = os.path.join(self.DATA_PATH, row['genre_name'], row['image_name'])
            image = Image.open(image_path)
            
            ax = axs[i // 5, i % 5]
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(row['genre_name'], fontsize=12)
        # plt.savefig(os.path.join(self.ASSETS_DIR, 'Examples of cover images.png'))
        plt.show()

        # Функция для получения размера изображения
        def get_image_size(image_path):
            image = Image.open(image_path)
            return image.size

        # Добавление колонки 'image_size' и заполнение размерами изображений
        dataset['image_size'] = dataset.apply(lambda row: get_image_size(os.path.join(self.DATA_PATH, row['genre_name'], row['image_name'])), axis=1)

        print(f"Количество полных повторов строк: {dataset.duplicated().sum()}")

        print(f"Количество повторов названий изображений: {dataset['image_name'].duplicated().sum()}")

        print(f"Размеры изображений: {dataset['image_size'].unique()}")

        return dataset


    def data_preprocessing(self):
        try:
            features_array = np.load(os.path.join(self.ASSETS_DIR, 'features.npy'))
        except FileNotFoundError:
            model_resnet = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
            model_resnet.eval()
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            features_list = []

            for genre in os.listdir(self.DATA_PATH):
                genre_path = os.path.join(self.DATA_PATH, genre)
                if os.path.isdir(genre_path):
                    for filename in os.listdir(genre_path):
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            image_path = os.path.join(genre_path, filename)
                            image = Image.open(image_path).convert('RGB')
                            image = transform(image).unsqueeze(0)

                            with torch.no_grad():
                                features = model_resnet(image)

                            features_np = features.numpy()
                            features_list.append(features_np)

            features_array = np.concatenate(features_list, axis=0)
            np.save(os.path.join(self.ASSETS_DIR, 'features.npy'), features_array)

        print(f"Размерность массива с векторами признаков: {features_array.shape}")

        return features_array

    def model_fitting(self, model_name=None, features=None, labels=None):
        X_tmp, X_test, y_tmp, y_test = train_test_split(features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
        X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tmp)

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_valid_encoded = label_encoder.transform(y_valid)
        y_test_encoded = label_encoder.transform(y_test)

        print("Размерности выборок:")
        print(f"обучающая {X_train.shape}")
        print(f"валидационная {X_valid.shape}")
        print(f"тестовая {X_test.shape}")

        if model_name == 'Baseline':
            model=None
            dimension = features.shape[1]
            index_faiss = faiss.IndexFlatL2(dimension)

            index_faiss.add(X_train.astype('float32'))
            k = 1  # Количество ближайших соседей для поиска

            D, I = index_faiss.search(X_valid.astype('float32'), k)

            # Получение предсказанных меток классов
            y_pred_encoded = y_train_encoded[I.flatten() % len(y_train_encoded)]
            
        elif model_name == 'SVM':
            model = SVC(kernel='linear', C=1.0)
            model.fit(X_train, y_train_encoded)
            y_pred_encoded = model.predict(X_valid)

        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        # расчёт метрик качества
        metrics = {}

        conf_matrix = confusion_matrix(y_valid, y_pred, normalize='all')
        precision =  precision_score(y_valid, y_pred, average='weighted')
        recall = recall_score(y_valid, y_pred, average='weighted')
        f1 = f1_score(y_valid, y_pred, average='weighted')
        accuracy = accuracy_score(y_valid, y_pred)
        err1 = conf_matrix[0, 1]
        err2 = conf_matrix[1, 0]

        # сохранение метрик в словарь
        metrics["precision"] = precision
        metrics["accuracy"] = accuracy            
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["err1"] = err1
        metrics["err2"] = err2

        valid_report = classification_report(y_valid, y_pred)
        print(f'Classification Report:\n{valid_report}')

        return metrics, X_train, y_train, X_test, y_test, model

    def model_logging(self,
					  experiment_name=None,
					  run_name=None,
					  registry_model=None,
					  params=None,
					  metrics=None,
					  model=None,
					  train_data=None,
					  train_label=None,
					  metadata=None,
					  code_paths=None,
					  tsh=None,
					  tsp=None):

        mlflow.set_tracking_uri(f"http://{tsh}:{tsp}")
        mlflow.set_registry_uri(f"http://{tsh}:{tsp}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id
		
        if run_name == 'baseline_0_registry':
            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                mlflow.log_metrics(metrics)
        else:
            pip_requirements = "requirements.txt"
            signature = mlflow.models.infer_signature(train_data, train_label.values)
            input_example = (pd.DataFrame(train_data)).iloc[0].to_dict()

            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                mlflow.log_artifacts(self.ASSETS_DIR)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="models",
                    pip_requirements=pip_requirements,
                    signature=signature,
                    input_example=input_example,
                    metadata=metadata,
                    code_paths=code_paths,
                    registered_model_name=registry_model,
                    await_registration_for=60
				)