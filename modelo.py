!unzip data_main.zip

!pip install optuna

def preprocess_metadata(metadata):
    numeric_cols = ['locationLon', 'locationLat', 'xps', 'Visits', 'Likes', 'Dislikes', 'Bookmarks']
    categorical_cols = ['categories']
    text_cols = ['name', 'shortDescription']

    for col in numeric_cols + categorical_cols + text_cols:
        if col not in metadata.columns:
            raise ValueError(f"Columna '{col}' no encontrada en el dataframe.")

    scaler = RobustScaler()
    numeric_scaled = scaler.fit_transform(metadata[numeric_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(metadata[categorical_cols])

    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
    text_data = metadata[text_cols].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    text_encoded = tfidf.fit_transform(text_data).toarray()

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(metadata['tags'])

    metadata_processed = np.hstack([numeric_scaled, categorical_encoded, text_encoded, tags_encoded])

    return metadata_processed

class POIDataset(Dataset):
def __init__(self, image_paths, metadata, labels, transform=None):
    self.image_paths = image_paths
    self.metadata = metadata
    self.labels = labels
    self.transform = transform

def __len__(self):
    return len(self.labels)

def __getitem__(self, idx):
    image = Image.open(self.image_paths[idx]).convert("RGB")
    if self.transform:
        image = self.transform(image)

    metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)
    label = torch.tensor(self.labels[idx], dtype=torch.float32)
    return image, metadata, label

class HybridModel(nn.Module):
    def __init__(self, metadata_input_size, dropout_rate=0.5, num_layers=2, neurons=[128, 64], freeze_image_model=False):
        super(HybridModel, self).__init__()
        self.image_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        image_features_dim = self.image_model.classifier[1].in_features  # Get in_features before replacing

        self.image_model.classifier = nn.Identity()  # Removemos la última capa de clasificación

        if freeze_image_model:
            for param in self.image_model.parameters():
                param.requires_grad = False

        self.metadata_layers = nn.ModuleList()
        input_dim = metadata_input_size

        for n in neurons:
            self.metadata_layers.append(nn.Linear(input_dim, n))
            self.metadata_layers.append(nn.ReLU())
            self.metadata_layers.append(nn.Dropout(dropout_rate))
            input_dim = n

        self.metadata_network = nn.Sequential(*self.metadata_layers)
        combined_input_size = image_features_dim + neurons[-1]
        self.fc_out = nn.Sequential(
            nn.Linear(combined_input_size, 1),
            nn.Sigmoid()
        )

    def forward(self, images, metadata):
        img_features = self.image_model(images)
        metadata_features = self.metadata_network(metadata)
        combined_features = torch.cat((img_features, metadata_features), dim=1)
        output = self.fc_out(combined_features)
        return output

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for images, metadata, labels in train_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, metadata).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct_train += ((outputs > 0.5).int() == labels.int()).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss / total_train)

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
                outputs = model(images, metadata).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                correct_val += ((outputs > 0.5).int() == labels.int()).sum().item()
                total_val += labels.size(0)

        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / total_val)
        print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_accuracies, val_accuracies, train_losses, val_losses



def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, metadata, labels in data_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs = model(images, metadata).squeeze()

            # Asegurarse de que outputs tenga la forma correcta
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "sensitivity": sensitivity,
        "specificity": specificity
    }
     


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, test_loss, test_accuracy):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='r', linestyle='-', label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
     

image_dir = 'data_main'
data_path = 'poi_dataset.csv'
threshold = 50
metadata = pd.read_csv(data_path)
image_paths, labels = [], []

for _, row in metadata.iterrows():
    image_path = row['main_image_path']
    if os.path.exists(image_path):
        image_paths.append(image_path)
        labels.append(1 if row['Likes'] - row['Dislikes'] > threshold else 0)

metadata_processed = preprocess_metadata(metadata)
train_paths, test_paths, train_metadata, test_metadata, train_labels, test_labels = train_test_split(
    image_paths, metadata_processed, labels, test_size=0.2, random_state=42
)
train_paths, val_paths, train_metadata, val_metadata, train_labels, val_labels = train_test_split(
    train_paths, train_metadata, train_labels, test_size=0.25, random_state=42
)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

common_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = POIDataset(train_paths, train_metadata, train_labels, transform=train_transform)
val_dataset = POIDataset(val_paths, val_metadata, val_labels, transform=common_transform)
test_dataset = POIDataset(test_paths, test_metadata, test_labels, transform=common_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(metadata_input_size=metadata_processed.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_accuracies, val_accuracies, train_losses, val_losses = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

model.eval()
test_loss, correct_test, total_test = 0.0, 0, 0
with torch.no_grad():
    for images, metadata, labels in test_loader:
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        outputs = model(images, metadata).squeeze()
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        correct_test += ((outputs > 0.5).int() == labels.int()).sum().item()
        total_test += labels.size(0)

test_accuracy = correct_test / total_test
metrics = evaluate_model(model, test_loader, device)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")