# Asesor Jurídico Educativo Ecuador — Streamlit

Chat público que responde preguntas sobre normativa educativa ecuatoriana, usando el grafo de
conocimiento de Graphify (`graph.json`, incluido en este paquete) como única fuente de verdad y
Claude (API de Anthropic) para redactar la respuesta citando las fuentes.

Este paquete es deliberadamente **liviano** (solo `graph.json`, ~340 KB) — no incluye los 56 PDFs
originales del corpus, para que el repo de GitHub y el despliegue en Streamlit sean rápidos. Los
PDFs siguen viviendo en tu proyecto local (`ASESOR_JURIDICO_EDUCATIVO_ECUADOR/`).

## 1. Probarlo en local (opcional pero recomendado antes de publicar)

```bash
cd streamlit_app
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edita `.streamlit/secrets.toml` y pon tu `ANTHROPIC_API_KEY` real (ver paso 2). Luego:

```bash
streamlit run streamlit_app.py
```

Se abre en `http://localhost:8501`.

## 2. Conseguir la API key de Anthropic (si no la tienes)

1. [console.anthropic.com](https://console.anthropic.com) → **API Keys** → **Create Key**.
2. **Billing** → agrega método de pago y carga crédito (unos $5–10 para empezar).
3. Guárdala — la usarás en el paso 4, nunca la subas a GitHub.

## 3. Subir este paquete a GitHub

Desde la carpeta `streamlit_app/`:

```bash
git init
git add .
git commit -m "Asesor Jurídico Educativo Ecuador - app Streamlit"
```

Crea el repo en GitHub (elige UNA opción):

**Opción A — con GitHub CLI (`gh`), si lo tienes instalado:**
```bash
gh repo create asesor-juridico-educativo-ecuador --public --source=. --remote=origin --push
```

**Opción B — manual:**
1. Ve a [github.com/new](https://github.com/new), crea un repo (ej. `asesor-juridico-educativo-ecuador`). Puede ser público o privado — Streamlit Cloud gratis solo despliega desde repos a los que tu cuenta tenga acceso (los privados funcionan si conectas tu cuenta de GitHub a Streamlit).
2. No marques "Initialize with README" (ya tienes uno).
3. Conecta y sube:
```bash
git remote add origin https://github.com/TU_USUARIO/asesor-juridico-educativo-ecuador.git
git branch -M main
git push -u origin main
```

## 4. Desplegar en Streamlit Community Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
2. **Create app** → **Deploy a public app from GitHub** (o el equivalente para repo privado).
3. Selecciona: repositorio `asesor-juridico-educativo-ecuador`, branch `main`, main file path
   `streamlit_app.py`.
4. Antes de dar "Deploy", abre **Advanced settings → Secrets** y pega:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-tu-clave-real"
   ANTHROPIC_MODEL = "claude-sonnet-5"
   MAX_QUESTIONS_PER_SESSION = 30
   ```
5. **Deploy**. En 1-2 minutos tendrás una URL pública tipo
   `https://asesor-juridico-educativo-ecuador.streamlit.app`.

## 5. Actualizar el grafo cuando agregues normativa nueva

En tu proyecto local, cada vez que agregues documentos:

```powershell
graphify "C:\Users\Usuario\Desktop\ASESOR_JURIDICO_EDUCATIVO_ECUADOR" --obsidian --update
```

Luego copia el `graph.json` actualizado a este paquete y sube los cambios:

```bash
copy ..\graphify-out\graph.json .\graph.json
git add graph.json
git commit -m "Actualizar grafo juridico"
git push
```

Streamlit Cloud redespliega automáticamente al detectar el push — no necesitas hacer nada más.

## Límites de esta v1

- **Sin historial persistente**: la conversación se guarda solo mientras la pestaña del navegador
  esté abierta (`st.session_state`); se pierde al recargar.
- **Límite por sesión, no por IP**: `MAX_QUESTIONS_PER_SESSION` limita preguntas por pestaña/sesión
  de Streamlit, no por usuario real — alguien podría abrir varias pestañas. Para un control más
  estricto haría falta autenticación o un proxy con rate-limit por IP.
- **Costo**: cada pregunta pública consume créditos de tu cuenta de Anthropic. Revisa el uso en
  [console.anthropic.com](https://console.anthropic.com) periódicamente.
