import cv2  # Biblioteca para processamento de imagens
import numpy as np  # Biblioteca para operações numéricas
import os  # Para verificar se o arquivo existe
import matplotlib.pyplot as plt  # Para exibir imagens
from skimage.feature import local_binary_pattern

# =====================================
# 1. Função para carregar imagem
# =====================================
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

# =====================================
# 2. Função para computar LBP
# =====================================
def compute_lbp(gray_region, n_points=24, radius=3):
    lbp = local_binary_pattern(gray_region, n_points, radius, method="uniform")
    bins = np.arange(0, n_points + 3)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normaliza o histograma
    return hist

# =====================================
# 3. Distância entre histogramas LBP
# =====================================
def chi_square_distance(histA, histB):
    dist = 0.0
    for i in range(len(histA)):
        if (histA[i] + histB[i]) > 0:
            dist += ((histA[i] - histB[i]) ** 2) / (histA[i] + histB[i])
    return dist

# =====================================
# 4. Carrega e prepara o histograma de referência (garrafa plástica)
# =====================================
def get_plastic_reference_hist(ref_image_path):
    ref_image = load_image(ref_image_path)
    if ref_image is None:
        print("Não foi possível carregar a imagem de referência.")
        return None
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_lbp = compute_lbp(ref_gray)
    return ref_lbp

# =====================================
# 5. Função principal para processar a imagem de resíduos utilizando Watershed e análise de textura
# =====================================
def process_waste_image(lixo_path, ref_path):
    # Carrega a imagem de referência (garrafa plástica) e calcula seu histograma LBP
    ref_hist = get_plastic_reference_hist(ref_path)
    if ref_hist is None:
        print("Erro ao obter histograma de referência.")
        return

    # Carrega a imagem do lixo
    image = load_image(lixo_path)
    if image is None:
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Segmentação com Threshold + Watershed
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    sure_bg = np.uint8(sure_bg)  # Garantir que sure_bg esteja em uint8
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    output = image.copy()
    output[markers == -1] = [0, 0, 255]

    object_descriptors = []
    unique_labels = np.unique(markers)
    obj_counter = 1

    for label in unique_labels:
        if label <= 1:
            continue
        mask = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # Calcular os Momentos de Hu (descritores de forma invariantes)
        hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()

        # Extrair a cor média do objeto utilizando uma máscara
        mask_obj = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask_obj, [cnt], -1, 255, -1)
        mean_bgr = cv2.mean(image, mask=mask_obj)[:3]

        # Calcular o descritor de textura usando LBP na região do objeto
        region = cv2.bitwise_and(gray, gray, mask=mask_obj)
        obj_lbp_hist = compute_lbp(region)

        # Comparar o histograma LBP do objeto com o de referência
        dist = chi_square_distance(obj_lbp_hist, ref_hist)
        threshold = 0.0905  # Limiar a ser ajustado conforme testes
        if dist < threshold:
            label_material = "Plastico"
        else:
            label_material = "Outro"

        # Desenhar o retângulo e o contorno do objeto na imagem de saída
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.drawContours(output, [cnt], -1, (255, 0, 0), 1)
        cv2.putText(output,
                    f"Objeto {obj_counter}",
                    (x, max(y-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # Texto em branco
                    2,
                    cv2.LINE_AA)

        # Armazenar todos os descritores em um dicionário
        object_info = {
            "id": obj_counter,
            "area": round(area, 2),
            "perimeter": round(perimeter, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "hu_moments": [round(val, 4) for val in hu_moments],
            "mean_bgr": tuple(map(int, mean_bgr)),
            "lbp_hist": obj_lbp_hist.tolist(),
            "lbp_dist_to_plastic": round(dist, 4),
            "material": label_material
        }
        object_descriptors.append(object_info)
        obj_counter += 1

    # Exibição dos resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Segmentação (Threshold + Morfologia)")
    plt.imshow(thresh_otsu, cmap='gray')
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Watershed + Detecção de Objetos")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"Objetos detectados: {len(object_descriptors)}")
    for obj_desc in object_descriptors:
        print(f"\nObjeto {obj_desc['id']}:")
        print(f" - Área           : {obj_desc['area']}")
        print(f" - Perímetro      : {obj_desc['perimeter']}")
        print(f" - Aspect Ratio   : {obj_desc['aspect_ratio']}")
        print(f" - Hu Moments     : {obj_desc['hu_moments']}")
        print(f" - Cor Média (BGR): {obj_desc['mean_bgr']}")
        print(f" - Histograma LBP : {obj_desc['lbp_hist']}")
        print(f" - Distância LBP para garrafa plástica: {obj_desc['lbp_dist_to_plastic']}")
        print(f" - Material estimado: {obj_desc['material']}")

# =====================================
# Execução do script
# =====================================
if __name__ == "__main__":
    # Caminho para a imagem de referência (garrafa plástica)
    ref_path = r"C:\Users\dred2\OneDrive\Imagens\Garrafa.jpg"
    # Caminho para a imagem do lixo
    lixo_path = r"C:\Users\dred2\OneDrive\Imagens\lixo.png"
    process_waste_image(lixo_path, ref_path)
