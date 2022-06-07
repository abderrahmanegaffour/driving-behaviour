mkdir -p ~/.streamlit/
echo "\
[theme]\n\
base = ’light’\n\
primaryColor = ‘#2696bd’\n\
secondaryBackgroundColor = ‘#ecebeb’\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

