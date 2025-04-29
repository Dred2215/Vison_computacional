import cv2
import numpy as np
import sys  # Para receber argumentos do terminal

# ===============================
# Função que calcula a cor média de um objeto usando máscara
# ===============================
def get_average_color(image, mask):
    mean_val = cv2.mean(image, mask=mask)  # Calcula média dos pixels dentro da máscara
    return mean_val[:3]  # Retorna apenas os canais BGR

# ===============================
# Função que classifica cor média em tipo de lixo reciclável
# ===============================
def classify_by_color(bgr_color):
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]  # Converte para HSV
    h, s, v = hsv_color

    if 0 <= h <= 10 or h >= 170:  # Vermelho
        return "Plástico"
    elif 20 <= h <= 35:  # Amarelo
        return "Metal"
    elif 40 <= h <= 85:  # Verde
        return "Vidro"
    elif 100 <= h <= 130:  # Azul
        return "Papel"
    elif 10 <= h <= 20 and s < 200:  # Marrom
        return "Orgânico"
    else:
        return "Desconhecido"

# ===============================
# Função principal que classifica as lixeiras por cor
# ===============================
def classify_waste_bins(image_path):
    image = cv2.imread(image_path)  # Lê a imagem do caminho informado
    if image is None:
        print("Erro ao carregar a imagem. Verifique o caminho.")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Converte para HSV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza

    # Aplica limiarização com Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontra os contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()  # Cópia da imagem original para desenhar resultados
    results = []  # Lista para guardar os dados dos objetos detectados

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue  # Ignora contornos pequenos

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h  # Calcula proporção largura/altura

        # Cria máscara para extrair a cor do objeto
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        avg_color = get_average_color(image, mask)

        # Classifica o tipo de lixo com base na cor
        waste_type = classify_by_color(avg_color)

        # Desenha na imagem
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, waste_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        # Armazena resultados
        results.append({
            "tipo": waste_type,
            "área": round(area, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "cor_média_BGR": tuple(map(int, avg_color))
        })

    # Exibe os resultados no terminal
    print("\nObjetos detectados e classificados:")
    for r in results:
        print(r)

    # Mostra a imagem com os resultados
    cv2.imshow("Resultado - Classificação de Lixeiras", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# Execução direta pelo script
# ===============================
# Substitui a execução por linha de terminal
image_path = r"C:\Users\dred2\OneDrive\Área de Trabalho\visão computacional\lixeiras-de-coleta-seletiva-larplasticos-1.jpg"
classify_waste_bins(image_path)

