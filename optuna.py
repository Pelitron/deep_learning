import optuna
import pandas as pd
import time
from torchvision.models import resnet18, inception_v3, vit_b_16
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_one_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
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

    train_auc = roc_auc_score(labels.cpu().numpy(), outputs.detach().cpu().numpy())

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

    val_auc = roc_auc_score(labels.cpu().numpy(), outputs.detach().cpu().numpy())

    # Update the scheduler with the validation loss
    scheduler.step(val_loss / total_val)

    return train_auc, val_auc
# Función objetivo para Optuna
def objective(trial, train_loader, val_loader, test_loader, device, metadata_size):
    """
    Función objetivo para búsqueda de hiperparámetros con Optuna.
    """
    # Selección de arquitectura de imagen
    architecture = trial.suggest_categorical("architecture", ["efficientnet_b0", "resnet18", "inception_v3", "vit_b_16"])
    if architecture == "efficientnet_b0":
        image_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    elif architecture == "resnet18":
        image_model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif architecture == "inception_v3":
        image_model = inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    elif architecture == "vit_b_16":
        image_model = vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # Congelar capas
    freeze_layers = trial.suggest_categorical("freeze_layers", [0, 5, 10])
    for i, param in enumerate(image_model.parameters()):
        param.requires_grad = i >= freeze_layers

    # Hiperparámetros del modelo híbrido
    neurons = [trial.suggest_int(f"neurons_{i}", 32, 256, step=32) for i in range(2)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6, step=0.1)
    bias = trial.suggest_categorical("bias", [True, False])

    # Crear modelo
    model = HybridModel(
        metadata_input_size=metadata_size,
        dropout_rate=dropout_rate,
        neurons=neurons,
        freeze_image_model=False
    ).to(device)

    # Selección de optimizador
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0, 1e-3, step=1e-4)
    momentum = trial.suggest_float("momentum", 0.5, 0.9) if optimizer_name == "SGD" else None

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Configurar criterio y scheduler
    criterion = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.1)

    # Entrenamiento y evaluación
    epochs = trial.suggest_int("epochs", 5, 20)
    best_val_auc = 0
    for epoch in range(epochs):
        train_auc, val_auc = train_one_epoch(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device
        )
        trial.report(val_auc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_val_auc = max(best_val_auc, val_auc)

    # Evaluación en test y métricas avanzadas
    start_time = time.time()
    metrics = evaluate_model(model, test_loader, device)
    inference_time = time.time() - start_time

    # Reportar métricas adicionales
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}, Specificity: {metrics['specificity']:.4f}")

    return metrics["auc"]

# Evaluación de modelo extendida
def evaluate_model(model, data_loader, device):
    """
    Evalúa el modelo y calcula métricas avanzadas.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, metadata, labels in data_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs = model(images, metadata).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calcular métricas
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    # Sensibilidad y especificidad
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

# Optimización con Optuna
def optimize_hyperparameters(train_loader, val_loader, test_loader, device, metadata_size, n_trials=50):
    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, test_loader, device, metadata_size),
        n_trials=n_trials
    )

    # Get the best trial
    best_trial = study.best_trial

    # Re-train the best model on the entire dataset
    model = HybridModel(
        metadata_input_size=metadata_size,
        dropout_rate=best_trial.params["dropout_rate"],
        neurons=[best_trial.params[f"neurons_{i}"] for i in range(2)],
        freeze_image_model=False
    ).to(device)

    optimizer_name = best_trial.params["optimizer"]
    lr = best_trial.params["lr"]
    weight_decay = best_trial.params["weight_decay"]
    momentum = best_trial.params.get("momentum", None)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.1)

    # Train the model on the entire dataset
    for epoch in range(best_trial.params["epochs"]):
        train_auc, val_auc = train_one_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    # Evaluate the model on the test set
    metrics = evaluate_model(model, test_loader, device)

    # Print and return the final metrics
    print(f"Final Metrics: {metrics}")

    # Save the study results
    study_results = pd.DataFrame(study.trials_dataframe())
    study_results.to_csv("optuna_advanced_results.csv", index=False)

    print("\nBest Trial:")
    print(study.best_trial)

    return study, metrics

# Ejecutar optimización
metadata_size = metadata_processed.shape[1]
study, final_metrics = optimize_hyperparameters(train_loader, val_loader, test_loader, device, metadata_size, n_trials=50)

# Visualización
plot_optimization_history(study).show()
plot_param_importances(study).show()

# Display final metrics
print(f"Final Metrics: {final_metrics}")