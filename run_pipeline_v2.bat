@echo off
echo ============================================
echo   EcoLab Pipeline V2 - Execution complete
echo ============================================
echo.

echo [1/8] Notation des articles avec Ollama (Mistral)...
echo       Necessite: Ollama avec mistral charge (ollama run mistral)
python 01_topic_notes_ollama_v2.py
if errorlevel 1 (echo ERREUR etape 1 & pause & exit /b 1)

echo.
echo [2/8] Visualisation des topics...
python 02_topic_viz_v2.py
if errorlevel 1 (echo ERREUR etape 2 & pause & exit /b 1)

echo.
echo [3/8] Score geospatial COSIA (long ~2h)...
python 03_geospatial_score_cosia_v2.py
if errorlevel 1 (echo ERREUR etape 3 & pause & exit /b 1)

echo.
echo [4/8] Enrichissement BD TOPO (long ~3h)...
python 04_enrich_cosia_bdtopo_v2.py
if errorlevel 1 (echo ERREUR etape 4 & pause & exit /b 1)

echo.
echo [5/8] Export QGIS (long ~4h)...
python 05_export_qgis_v2.py
if errorlevel 1 (echo ERREUR etape 5 & pause & exit /b 1)

echo.
echo [6/8] Articles par parcelle...
python 06_articles_par_parcelle_v2.py
if errorlevel 1 (echo ERREUR etape 6 & pause & exit /b 1)

echo.
echo [7/8] Resumes BD TOPO + injection...
python 06b_resume_bdtopo.py
if errorlevel 1 (echo ERREUR etape 7a & pause & exit /b 1)
python 06c_inject_resume_bdtopo.py
if errorlevel 1 (echo ERREUR etape 7b & pause & exit /b 1)

echo.
echo [8/8] Creation des index SQLite...
python create_indexes.py
if errorlevel 1 (echo ERREUR etape 8 & pause & exit /b 1)

echo.
echo ============================================
echo   Pipeline terminee avec succes!
echo   Lancer le Streamlit: streamlit run app_carte2.py
echo ============================================
pause
