# DroneVision
A simple computer vision system for drones that identifies objects and follows paths.

### About
This is a project made by <a href="https://github.com/Sunref">Sunref</a> and <a href="https://github.com/gm64x">gm64x</a> for the Computer Vision class.

## Description

This project implements computer vision algorithms on a drone to:
* Detect and classify objects in real-time
* Identify paths in the environment
* Navigate autonomously by following these paths

The system processes video feed from the drone's camera to recognize objects such as vehicles, buildings, people, and obstacles. Using edge detection and path recognition algorithms, it can identify walkways, roads, or designated routes and guide the drone to follow them while avoiding detected obstacles.

# Structure
├── data          # Armazena os dados do projeto, como conjuntos de dados, dados brutos e processados
├── docs          # Documentação do projeto, incluindo especificações técnicas e manuais de usuário
├── notebooks     # Notebooks Jupyter/Python para análises exploratórias e demonstrações
├── reports       # Relatórios gerados, resultados e saídas do projeto
│   └── images    # Imagens usadas nos relatórios ou geradas pelo projeto
├── src           # Código-fonte do projeto, contendo os módulos e classes principais
└── tests         # Testes unitários e de integração para verificar a funcionalidade do código
