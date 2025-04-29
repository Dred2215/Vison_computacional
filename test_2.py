import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path, resize_dim=(800, 800)):
    """Carrega e redimensiona a imagem."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Erro ao carregar a imagem. Verifique o caminho.")
    return cv2.resize(img, resize_dim)


def apply_floodfill(img):
    """Aplica a técnica de FloodFill para segmentação da vegetação."""
    img_flood = img.copy()
    mask_flood = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    seed_points = [(100, 100), (400, 400), (700, 700)]
    flood_flags = 4 | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    for seed in seed_points:
        cv2.floodFill(img_flood, mask_flood, seed, (0, 255, 0), (200, 200, 200), (100, 100, 20), flood_flags)
    return img_flood

def apply_morphological_processing(img):
    """Aplica operações morfológicas para suavizar bordas e remover riscos pretos."""
    kernel = np.ones((5, 5), np.uint8)
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Fecha buracos e suaviza bordas
    return img_closed


def display_results(original, floodfill_segmented, floodfill_refined):
    """Exibe todas as imagens processadas lado a lado."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(floodfill_segmented, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Segmentação - Apenas FloodFill")
    axes[1].axis("off")
    
       
    axes[2].imshow(cv2.cvtColor(floodfill_refined, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Segmentação - FloodFill Refinado")
    axes[2].axis("off")
    
    plt.show()
    
# Executando o fluxo completo
image_path = "C:/Users/dred2/Downloads/Visao_computacional/DESMATAMENTO.jpg"
img = load_image(image_path)
img_flood = apply_floodfill(img)
img_flood_refined = apply_morphological_processing(img_flood)

display_results(img, img_flood, img_flood_refined)
