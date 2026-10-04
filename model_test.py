import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Arayüz dosyanızdan model sınıfını içe aktarın (Dosya adınız neyse 'arayuz_dosyasi' kısmını ona göre değiştirin)
# Eğer model sınıfını import edemezseniz, ScalableDiagramCNN sınıfının kodunu doğrudan bu dosyanın en üstüne kopyalayabilirsiniz.
from edge_ai_train_model_gui_quantization_according_succes_rate import ScalableDiagramCNN 

# 1. Test verisini yükle
try:
    data = np.load('bldc_dataset.npz')
    X_test_t = torch.tensor(data['X_test'], dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_t = torch.tensor(data['y_test'], dtype=torch.long)
except FileNotFoundError:
    print("HATA: bldc_dataset.npz bulunamadı.")
    exit()

# 2. NAS'ın bulduğu optimum parametreler (Görsele göre: 9 Kanal, 32x13 giriş)
OPTIMUM_KANAL = 9 
model = ScalableDiagramCNN(channels=OPTIMUM_KANAL, input_h=32, input_w=13, num_classes=3, num_layers=2)

# 3. Eğitilmiş ağırlıkları yükle
try:
    model.load_state_dict(torch.load(f'./export/edge_cnn_int16_{OPTIMUM_KANAL}ch.pth'))
    model.eval()
    print(f"{OPTIMUM_KANAL} Kanallı Model Başarıyla Yüklendi!\n")
except Exception as e:
    print(f"Ağırlıklar yüklenirken hata oluştu: {e}")
    exit()

# 4. Tüm test seti üzerinde tahmin yap
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)

# 5. Raporu Çıkar
y_true = y_test_t.numpy()
y_pred = predicted.numpy()

print("--- MODEL KARNESİ (SINIFLANDIRMA RAPORU) ---")
print(classification_report(y_true, y_pred, target_names=["Sağlıklı", "Rulman Arızası", "Pervane Arızası"]))

print("\n--- KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) ---")
print("Satırlar: Gerçek Değerler | Sütunlar: Tahmin Edilen Değerler")
print(confusion_matrix(y_true, y_pred))