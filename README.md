# TFG: Sistema de detección, explicación y visualización de comunidades

## Instalar Python y JupyterLab
(JupyterLab es solo necesario si se desea usar la interfaz gráfica)
### Windows
- Instalar Python desde: https://www.python.org/downloads/
- Instalar JupyterLab:
  - Abrir terminal:
    - Tecla Windows + R
    - Escribir: cmd
    - Pulsar Enter
  - Escribir: `pip install jupyterlab`

- **Alternativa**: Usar Anaconda:
  - Instalar Anaconda desde: https://docs.anaconda.com/anaconda/install/windows/

### Unix
```bash
sudo apt-get install python
pip install jupyterlab
```
## Requisitos externos
` pip install -r requirements.txt `

## Interfaz gráfica
- Ejecutar JupyterLab desde terminal: `jupyter lab`
- Abrir [`Interfaz.ipynb`](https://github.com/AlbertoGarciaDomenech/Comunidades-en-museos/blob/main/Interfaz.ipynb)
- Instalar requisitos (solo la primera vez): Ejecutar primera celda de código
- Usar interfaz gráfica: Ejecutar segunda celda de código y seguir los pasos
  - Los datos de prueba del Museo del Prado ya están en su correspondiente sitio
  - Para usar nuevos datos, los datos iniciales deben estar en _/data/_ y las matrices de similitud pre-generadas en _/data/sim/_
