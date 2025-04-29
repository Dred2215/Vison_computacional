#!/usr/bin/env bash   # Define o interpretador do script como bash

# Script para exibir o README do projeto de Visão Computacional

cat << 'EOF'           # Inicia um bloco de texto multilinha
# Visão Computacional

## Descrição
Este projeto reúne dois pipelines de visão computacional desenvolvidos para fins acadêmicos:
1. **pbl1** – Detecção de desmatamento  
2. **pbl2** – Classificação de resíduos sólidos  

Cada pipeline está isolado em sua própria pasta, com scripts completos e exemplos de entrada.

---

## Estrutura do Repositório
VISAO_COMPUTACIONAL/                    # Pasta raiz do projeto
├─ pbl1/                               # Pipeline 1: Detecção de desmatamento
│   ├─ DESMATAMENTO2.jpg               # Imagem de exemplo
│   ├─ modelos_usados_de_base.py       # Funções auxiliares
│   ├─ identificacao.py                # Segmentação via FloodFill
│   ├─ main.py                         # Orquestrador do pipeline
│   └─ test_2.py                       # Script de testes
├─ pbl2/                               # Pipeline 2: Classificação de resíduos
│   ├─ 20250224_00_lixo_residuos_solidos.jpg  # Imagem de exemplo
│   ├─ Garrafa.jpg                     # Imagem de referência (plástico)
│   ├─ identificar_lixo.py             # Segmentação + LBP + Watershed
│   ├─ identificar_lixo_2.py           # Versão estendida com classificação por cor
│   └─ lixeiras-de-coleta-seletiva-larplasticos-1.jpg  # Imagem de coletores
├─ requirements.txt                    # Dependências Python
└─ README.md                           # Este documento

---

## Pré-requisitos
- Python 3.8 ou superior  
- OpenCV  
- NumPy  
- Matplotlib  
- scikit-image  

> **Dica:** Use um _virtual environment_ para não poluir seu ambiente global.

---

## Instalação
```bash
python -m venv venv               # cria ambiente virtual  
source venv/bin/activate          # ativa no Linux/macOS  
venv\Scripts\activate             # ativa no Windows  
pip install -r requirements.txt   # instala dependências  
