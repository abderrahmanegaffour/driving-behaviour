mkdir -p ~/.streamlit/
echo "[theme]
base = ’light’
primaryColor = ‘#2696bd’
backgroundColor = ‘#EFEDE8’
secondaryBackgroundColor = ‘#ecebeb’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
