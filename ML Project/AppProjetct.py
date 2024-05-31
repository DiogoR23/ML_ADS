import streamlit as st

st.title("Minha Primeira Aplicação com Streamlit")
st.write("Olá, mundo! Esta é a minha aplicação web criada com Streamlit.")

nome = st.text_input("Digite seu nome:")
st.write(f"Olá, {nome}!")

if st.button("Clique aqui"):
    st.write("Você clicou no botão!")