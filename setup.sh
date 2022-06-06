mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"a.gaffour@esi-sba.dz\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
