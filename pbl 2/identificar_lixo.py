import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ===============================
# Função para carregar uma imagem JPG usando imdecode
# ===============================
def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Erro: O arquivo '{image_path}' não foi encontrado.")
        return None
    try:
        with open(image_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            print("Erro: A imagem não pôde ser decodificada. Verifique o formato e integridade do arquivo.")
        return image
    except Exception as e:
        print("Erro ao abrir o arquivo:", e)
        return None

# ===============================
# Função que classifica a cor média em tipo de lixo reciclável
# ===============================
def classify_by_color(bgr_color):
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color

    # Ajuste fino dos intervalos de cor pode ser necessário.
    if 0 <= h <= 10 or h >= 170:  
        return "Plástico"   # Tons de vermelho
    elif 20 <= h <= 35:     
        return "Metal"      # Tons de amarelo
    elif 40 <= h <= 85:     
        return "Vidro"      # Tons de verde
    elif 100 <= h <= 130:   
        return "Papel"      # Tons de azul
    elif 10 <= h <= 20 and s < 200:
        return "Orgânico"   # Tons de marrom
    else:
        return "Desconhecido"

# ===============================
# Função para processar cada ROI (lixeira) individualmente
# ===============================
def process_bins_by_rois(image_path, rois):
    image = load_image(image_path)
    if image is None:
        return
    
    height, width, _ = image.shape
    print(f"Dimensões da imagem: {width}x{height} (LxA)\n")

    results = []
    fig, axs = plt.subplots(1, len(rois), figsize=(15, 5))
    if len(rois) == 1:
        axs = [axs]

    for idx, (bin_label, (x, y, w, h)) in enumerate(rois.items()):
        # Verifica e ajusta as coordenadas para não exceder os limites
        if x < 0 or y < 0 or x >= width or y >= height:
            print(f"ROI para '{bin_label}' possui coordenadas inválidas: x={x}, y={y}")
            continue
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y

        roi = image[y:y+h, x:x+w]
        print(f"{bin_label}: ROI shape = {roi.shape}, coords = ({x}, {y}, {w}, {h})")

        if roi.size == 0:
            print(f" -> ROI vazia. Verifique se as coordenadas de '{bin_label}' estão corretas.")
            continue

        # Cor média BGR na ROI
        avg_color = cv2.mean(roi)[:3]
        predicted_label = classify_by_color(avg_color)

        results.append({
            "lixeira": bin_label,
            "tipo": predicted_label,
            "cor_média_BGR": tuple(map(int, avg_color))
        })

        axs[idx].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axs[idx].set_title(f"{bin_label}\nPred: {predicted_label}")
        axs[idx].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nResultados Finais:")
    for r in results:
        print(r)

# ===============================
# Execução direta pelo script
# ===============================
if __name__ == "__main__":
    # Ajuste o caminho da sua imagem aqui
    image_path = r"C:\Users\dred2\OneDrive\Área de Trabalho\vision_trash\lixeiras-de-coleta-seletiva-larplasticos-1.jpg"
    
    # ROIs ajustadas para uma imagem de ~700-800 px de largura e ~400-500 px de altura
    # Se não estiver correto, ajuste (x, y, w, h) manualmente até "enquadrar" cada lixeira
    rois = {
        # bin_label: (x, y, w, h)
        "Vermelho": (10,  90, 130, 300),   # Ajuste se necessário
        "Azul": (150, 90, 130, 300),
        "Verde": (290, 90, 130, 300),
        "Marrom": (430, 90, 130, 300),
        "Amarelo": (570, 90, 130, 300)
    }
    
    process_bins_by_rois(image_path, rois)
