import streamlit as st
import pandas as pd
import pickle
import lzma
from sklearn.preprocessing import LabelEncoder
import traceback
import os
from datetime import datetime
import numpy as np
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="ANVISA - Sistema de Fiscalização de Produtos",
    page_icon="🛡️",
    layout="wide"
)

def header_anvisa(): # Adiciona o cabeçalho com informações do governo
    st.markdown("""
    <style>
        .gov-header {
            background-color: #006341; /* Verde ANVISA */
            padding: 0.5rem 1rem;
            color: white;
            font-family: 'Arial', sans-serif;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .gov-logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .gov-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }
        .gov-subtitle {
            font-size: 1rem;
            margin: 0;
            opacity: 0.9;
        }
        .gov-brasil {
            background-color: #072b57; /* Azul governo */
            padding: 0.3rem;
            font-size: 0.9rem;
            display: flex;
            justify-content: center;
            gap: 20px;
        }
    </style>

    <div class="gov-brasil">
        <span>🌐 brasil.gov.br</span>
        <span>🛡️ Ministério da Saúde</span>
        <span>🇧🇷 Governo Federal</span>
    </div>
    """, unsafe_allow_html=True)
header_anvisa()

# Função para verificar arquivos 
def check_files():
    required_files = [
        'data/le_categoria.pkl', 
        'data/le_empresa.pkl',
        'data/le_produto.pkl',
        'data/le_target.pkl',
        'data/modelo_final.pkl.xz',
        'data/produtos_classificados.csv'
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        st.error(f"Arquivos faltando: {', '.join(missing)}")
        return False
    return True

# Função para carregar LabelEncoders
def load_label_encoder(filepath):
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            
            # Caso 1: Já é um LabelEncoder
            if isinstance(obj, LabelEncoder):
                return obj
                
            # Caso 2: É um array numpy com as classes
            encoder = LabelEncoder()
            encoder.classes_ = obj
            return encoder
            
    except Exception as e:
        st.error(f"Erro ao carregar {filepath}: {str(e)}") 
        traceback.print_exc()
        return None

# Carregar todos os dados
@st.cache_resource
def load_data():
    if not check_files():
        return None
        
    try:
        # Carregar encoders
        le_categoria = load_label_encoder('data/le_categoria.pkl')
        le_empresa = load_label_encoder('data/le_empresa.pkl')
        le_produto = load_label_encoder('data/le_produto.pkl')
        le_target = load_label_encoder('data/le_target.pkl')
        
        # Verificar se todos os encoders foram carregados
        if None in [le_categoria, le_empresa, le_produto, le_target]:
            return None
            
        # Carregar modelo
        with lzma.open('data/modelo_final.pkl.xz', 'rb') as f:
            modelo = pickle.load(f)
            
        # Carregar dados
        produtos_df = pd.read_csv('data/produtos_classificados.csv', sep=',')
        
        return {
            'le_categoria': le_categoria,
            'le_empresa': le_empresa,
            'le_produto': le_produto,
            'le_target': le_target,
            'modelo': modelo,
            'produtos_df': produtos_df
        }
        
    except Exception as e:
        st.error(f"Erro fatal ao carregar dados: {str(e)}")
        traceback.print_exc()
        return None
    

data = load_data()

# Função para adicionar logo e estilo
def add_logo():
    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .css-18e3th9 {padding-top: 0rem;}
            
            /* Cores da ANVISA */
            :root {
                --anvisa-green: #006341;
                --anvisa-light-green: #e1f0e8;
                --anvisa-blue: #005b8c;
            }
            
            .stButton>button {
                background-color: var(--anvisa-green);
                color: white;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                border: none;
            }
            
            .stButton>button:hover {
                background-color: var(--anvisa-blue);
                color: white;
            }
            
            .css-1aumxhk {
                background-color: var(--anvisa-light-green);
                border-radius: 10px;
                padding: 20px;
            }
            
            .css-1v0mbdj {
                border-radius: 10px;
            }
            
            .tab {
                font-size: 18px;
                font-weight: bold;
            }
            
            h1 {
                color: var(--anvisa-green);
                border-bottom: 2px solid var(--anvisa-green);
                padding-bottom: 10px;
            }
            
            h2 {
                color: var(--anvisa-blue);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Logo ANVISA
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <h1 style="color: #006341; margin: 0; padding: 0;">ANVISA</h1>
            <p style="margin-left: 10px; color: #555; font-style: italic;">Sistema de Fiscalização de Produtos</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Função para fazer previsões
def predict_product(categoria, produto, empresa):
    try:
        # Buscar informações do produto no dataframe
        product_info = data['produtos_df'][
            (data['produtos_df']['DS_CATEGORIA_PRODUTO'] == categoria) & 
            (data['produtos_df']['NO_PRODUTO'] == produto) & 
            (data['produtos_df']['NO_RAZAO_SOCIAL_EMPRESA'] == empresa)
        ].iloc[0]

        # Aplicar a lógica de classificação manualmente
        situacao = product_info['ST_SITUACAO_REGISTRO']
        vencimento = pd.to_datetime(product_info['DT_VENCIMENTO_REGISTRO'])
        dias_para_vencer = (vencimento - pd.to_datetime('today')).days

        if situacao == 'INATIVO':
            classificacao = 'INATIVO'
        elif situacao == 'ATIVO':
            if dias_para_vencer < 0:
                classificacao = 'VENCIDO'
            elif 0 <= dias_para_vencer <= 180:
                classificacao = 'PERTO DO VENCIMENTO'
            else:
                classificacao = 'ATIVO'
        else:
            classificacao = 'INDEFINIDO'

        return {
            'classificacao': classificacao, 
            'validade': product_info['DT_VENCIMENTO_REGISTRO'],
            'empresa': product_info['NO_RAZAO_SOCIAL_EMPRESA'],
            'registro': product_info['NU_REGISTRO_PRODUTO']
        }
    except Exception as e:
        st.error(f"Erro ao fazer previsão: {str(e)}")
        return None

# Página principal
def main():
    add_logo()
    
    if data is None:
        st.error("""
        **Falha crítica no carregamento de dados.**
        
        Por favor, verifique:
        1. Todos os arquivos necessários estão na pasta
        2. Os arquivos .pkl não estão corrompidos
        3. O arquivo CSV está no formato correto
        
        Arquivos necessários:
        - le_categoria.pkl, le_empresa.pkl, le_produto.pkl, le_target.pkl
        - modelo_final.pkl.xz
        - produtos_classificados.csv
        """)
        return
    
    tab1, tab2 = st.tabs(["Conheça o Projeto", "Consulta de Produtos"])
    
    with tab1:
        st.header("📊 Sobre a Base")
    
            # Container com estatísticas
        st.markdown("""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px;">
                <div style="background: #006341; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <h3 style="color: #e1f0e8; margin: 0;">✅ Ativos</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">36%</p>
                </div>
                <div style="background: #d4a017; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <h3 style="color: #fff8e6; margin: 0;">⏳ Perto do Venc.</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">2%</p>
                </div>
                <div style="background: #d32f2f; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <h3 style="color: #ffebee; margin: 0;">❌ Vencidos</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">9,5%</p>
                </div>
                <div style="background: #616161; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <h3 style="color: #f5f5f5; margin: 0;">🔒 Inativos</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">52,5%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Seção "Objetivo do Projeto" 
        with st.expander("🎯 **Objetivo do Projeto**", expanded=True):
                st.markdown("""
                Este sistema foi desenvolvido para fiscais da ANVISA, objetivando auxiliar na classificação de produtos regulamentados, 
                agilizando o processo de fiscalização utilizando apenas alguns cliques, garantindo maior consistência nas informações e na 
                tomada de decisão. Na classificação os produtos são divididos em **4 categorias**:
                
                - **✅ ATIVO**: Produto dentro do prazo de validade e sem restrições.
                - **⏳ PERTO DO VENCIMENTO**: Validade expira em até 180 dias.
                - **❌ VENCIDO**: Data de validade ultrapassada.
                - **🔒 INATIVO**: Registro cancelado ou suspenso pela ANVISA.
                
                Os cards acima mostram a porcentagem de produtos em cada categoria, com base nos dados disponíveis.
                """)
                
                st.markdown("""
                ### Base Legal
                Segundo a **NIT/Dicla-035**, norma que define os princípios das Boas Práticas de Laboratório (BPL), é obrigatória a rotulagem de produtos químicos, reagentes e soluções, contendo informações como:
                
                - Identidade do produto
                - Orientações de armazenamento
                - Prazo de validade *(INMETRO, 2011)*
                            
                Conforme a RDC nº 157/2002 da ANVISA, produtos com registro vencido ou irregular 
                devem ser imediatamente retirados do mercado.
                
                > *"O prazo de validade corresponde ao período em que o produto permanece adequado para uso, também denominado vida útil, sendo determinado com base em estudos de estabilidade específicos"* **(BRASIL, 2002)**.
                """)

            # Seção do Modelo
        with st.expander("📈 **Sobre o Modelo**"):
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.markdown("""
                    <div style='background-color: #006341; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1)'>
                        <h3 style='color: white; margin: 0;'>Acurácia do Modelo</h3>
                        <p style='font-size: 32px; font-weight: bold; color: white; margin: 10px 0;'>97.43%</p>
                        <p style='color: #e1f0e8; margin: 0;'>Taxa de classificação correta</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Análise
                st.subheader("📌 Análise da Resultados")
                st.markdown("""
                Foram testados cinco modelos de aprendizagem supervisionada, utilizando validação cruzada com 10 folds. 
                As métricas de avaliação incluíram acurácia média, desvio padrão e matriz de confusão.
                            
                Após se mostrar o modelo com melhor desempenho, o Random Forest foi escolhido como benchmark (modelo principal), por apresentar o melhor equilíbrio entre desempenho geral (acurácia de 61,0%), 
                robustez em ambientes com ruídos e capacidade de interpretar variáveis categóricas complexas.
                Além disso, o modelo foi excelente na realização de um novo treinamento com 
                o conjunto de dados completo e balanceado, atingindo os seguintes valores na matriz de confusão, dando destaque para:
                            
                """)
                st.markdown("""
                
                - **Alta precisão em classificar registros INATIVOS** (98% de acertos) e ATIVOS (97% de acertos)
                - **Bom reconhecimento de produtos VENCIDOS** (98% de recall), crucial para fiscalização
                - **Desafio com a classe minoritária (PERTO DO VENCIMENTO)**:
                    - Apesar do alto recall (97%), a precisão foi menor (68%)
                    - Isso indica que o modelo às vezes classifica outros casos como "Perto do Vencimento" erroneamente
                - **Principais erros**: Confusão entre:
                    - INATIVO classificado como VENCIDO (216 casos)
                    - ATIVO classificado como VENCIDO (156 casos)
                """)
                
                # Tabela de Métricas
                st.subheader("📈 Métricas Detalhadas por Classe")
                
                metrics_df = pd.DataFrame({
                    'Classe': ['ATIVO', 'INATIVO', 'PERTO DO VENCIMENTO', 'VENCIDO'],
                    'Precisão': [0.99, 1.00, 0.68, 0.90],
                    'Recall': [0.97, 0.98, 0.97, 0.98],
                    'F1-Score': [0.98, 0.99, 0.80, 0.94],
                    'Exemplos': [10569, 15367, 559, 2796]
                })
                
                # Formatação condicional para destacar valores baixos
                def color_low(val):
                    color = 'red' if val < 0.7 else 'green' if val > 0.9 else 'orange'
                    return f'color: {color}; font-weight: bold'
                
                styled_df = metrics_df.style.applymap(color_low, subset=['Precisão', 'Recall', 'F1-Score'])
                
                st.dataframe(
                    styled_df.format({
                        'Precisão': '{:.2%}',
                        'Recall': '{:.2%}',
                        'F1-Score': '{:.2%}'
                    }),
                    hide_index=True,
                    use_container_width=True
                )
                
                # Explicação complementar
        with st.expander("ℹ️ Como interpretar estas métricas?"):
                st.markdown("""
                    - **Precisão**: De todos que o modelo classificou como X, quantos realmente eram X?
                        - *Exemplo: 68% dos classificados como "Perto do Vencimento" estavam corretos*
                    - **Recall**: De todos os casos reais de X, quantos o modelo identificou?
                        - *Exemplo: 97% dos produtos realmente perto do vencimento foram detectados*
                    - **F1-Score**: Média harmônica entre Precisão e Recall (quanto maior, melhor)
                    """)
            
            # Seção de Limitações
        with st.expander("⚠️ **Limitações e Recomendações**"):
                st.warning("""
                - Recomenda-se **atualização trimestral** do modelo com novos dados de fiscalização.
                - Incorporar variáveis adicionais, como histórico de fiscalização ou reclamações de consumidores, para enriquecimento do modelo.
                - Explorar técnicas avançadas de ensemble, como Gradient Boosting, para potencializar o desempenho.
                - Implementar o modelo em sistemas de monitoramento automatizado para auxiliar órgãos reguladores e empresas do setor.
                - Considerar a implementação de um sistema de feedback contínuo, onde os usuários possam reportar erros ou inconsistências nas previsões, permitindo ajustes e melhorias no modelo ao longo do tempo.
                """)


    with tab2:
        st.header("🔍 Sistema de Fiscalização Automatizada ANVISA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Verificar se os dados foram carregados corretamente
                if data is None:
                    st.error("Dados não carregados corretamente. Verifique os arquivos necessários.")
                    return
                
                # Seleção da categoria
                categoria = st.selectbox(
                    "Selecione a Categoria do Produto",
                    options=data['le_categoria'].classes_,
                    index=0
                )

                # Filtrar produtos baseado na categoria selecionada
                produtos_filtrados = data['produtos_df'][
                    data['produtos_df']['DS_CATEGORIA_PRODUTO'] == categoria
                ]['NO_PRODUTO'].unique()

                # Seleção do produto
                produto = st.selectbox(
                    "Selecione o Produto",
                    options=produtos_filtrados,
                    index=0
                )

                # Filtrar empresas baseado no produto selecionado
                empresas_filtradas = data['produtos_df'][
                    (data['produtos_df']['DS_CATEGORIA_PRODUTO'] == categoria) &
                    (data['produtos_df']['NO_PRODUTO'] == produto)
                ]['NO_RAZAO_SOCIAL_EMPRESA'].unique()

                # Seleção da empresa
                empresa = st.selectbox(
                    "Selecione a Empresa",
                    options=empresas_filtradas,
                    index=0
                )

                if st.button("Consultar Produto"):
                    with st.spinner("Processando consulta..."):
                        resultado = predict_product(categoria, produto, empresa)  
                
                        if len(empresas_filtradas) == 0:
                            st.warning("Nenhuma empresa encontrada para este produto/categoria.")
                            return
                                        
                        if resultado:
                            # Mensagens condicionais baseadas na classificação
                            if resultado['classificacao'] == "PERTO DO VENCIMENTO":
                                st.warning("⚠️ **Atenção:** Produto perto do vencimento. Ação recomendada em 180 dias conforme RDC 157/2002.")
                            elif resultado['classificacao'] == "ATIVO":
                                st.success("✅ Classificação confirmada conforme legislação ANVISA.")
                            elif resultado['classificacao'] == "VENCIDO":
                                st.error("❌ **Produto vencido:** Retirada imediata do mercado exigida pela legislação.")
                            elif resultado['classificacao'] == "INATIVO":
                                st.info("ℹ️ **Registro inativo:** Verificar motivo da inativação no sistema ANVISA.")
                            st.session_state['mostrar_formulario'] = True  # Ativa flag na sessão

                            # Exibir resultados em cards
                            st.markdown("### Resultado da Consulta")
                                    
                            col_res1, col_res2 = st.columns(2)
                            
                            with col_res1:
                                st.markdown(
                                    f"""
                                    <div style='background-color: #006341; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                                        <h4 style='color: #e1f0e8; margin-top: 0;'>Classificação</h4>
                                        <p style='font-size: 18px;'>{resultado['classificacao']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                st.markdown(
                                    f"""
                                    <div style='background-color: #006341; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                                        <h4 style='color: #e1f0e8; margin-top: 0;'>Empresa Responsável</h4>
                                        <p style='font-size: 18px;'>{resultado['empresa']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with col_res2:
                                st.markdown(
                                    f"""
                                    <div style='background-color:  #006341; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                                        <h4 style='color: #e1f0e8; margin-top: 0;'>Data de Validade</h4>
                                        <p style='font-size: 18px;'>{resultado['validade']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                st.markdown(
                                    f"""
                                    <div style='background-color: #006341; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                                        <h4 style='color: #e1f0e8; margin-top: 0;'>Número de Registro</h4>
                                        <p style='font-size: 18px;'>{resultado['registro']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Formulário INDEPENDENTE para report de erro
                            if st.session_state.get('mostrar_formulario', False):
                                with st.form(key='feedback_form'):
                                    st.subheader("✏️ Reportar Erro")
                                    erro = st.text_area("Use está seção apenas em caso de identificação de erro na classificação. Descreva detalhadamente o problema encontrado:")
                                    
                                    if st.form_submit_button("Enviar Relatório"):
                                        if erro.strip():
                                            # Salvar feedback
                                            feedback = {
                                                'data': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                'produto': produto,
                                                'empresa': empresa,
                                                'erro': erro
                                            }
                                            pd.DataFrame([feedback]).to_csv(
                                                "feedbacks.csv",
                                                mode='a',
                                                header=not os.path.exists("feedbacks.csv"),
                                                index=False
                                            )
                                            st.success("✅ Relatório enviado à equipe ANVISA!")
                                            st.session_state['mostrar_formulario'] = False  # Fecha formulário
                                        else:
                                            st.warning("Por favor, descreva o erro encontrado.")
                                st.markdown("""
                                <style>
                                    div[data-testid="stForm"] {
                                        border: 1px solid #006341;
                                        border-radius: 10px;
                                        padding: 20px;
                                    }
                                </style>
                                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erro ao carregar opções de consulta: {str(e)}")
                traceback.print_exc()
                            
        with col2:
            st.markdown("### Informações Adicionais")
            st.markdown("""
            Este sistema utiliza inteligência artificial para classificar produtos regulamentados em quatro categorias críticas:

            - ✅ ATIVO - Registros válidos e dentro do prazo
            - ⚠️ PERTO DO VENCIMENTO - Validade expira em até 180 dias
            - ❌ VENCIDO - Registros com prazo expirado
            - 🔒 INATIVO - Registros cancelados ou suspensos
                        
            Como funciona:
            - Selecione categoria, produto e empresa
            - Obtenha a classificação regulatória instantânea
            - Acesse informações completas sobre validade e status
            
            *Baseado no modelo de Machine Learning com 97.43% de acurácia, conforme Portaria ANVISA nº 157/2002*
            """)
            
            st.markdown("""
            **Dúvidas?** Entre em contato com a equipe de TI da ANVISA:
            - Email: ti@anvisa.gov.br
            - Telefone: (61) 3462-5400
            """)
            # RODAPÉ
            st.markdown("""
            <style>
            /* Evita conflito com abas */
            .stApp [data-testid="stVerticalBlock"] {
                position: relative;
                padding-bottom: 50px !important; /* Espaço para o rodapé */
            }
            
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #006341;
                color: white;
                text-align: center;
                padding: 8px 0;
                font-size: 0.85rem;
                z-index: 999;
                border-top: 1px solid #e1f0e8;
            }
            
            .footer-content {
                max-width: 800px;
                margin: 0 auto;
                display: flex;
                justify-content: center;
                gap: 15px;
                flex-wrap: wrap;
            }
            </style>
            
            <div class="footer">
                <div class="footer-content">
                    <span>Projeto por: <strong>Lívia Fernandes da Rocha</strong></span>
                    <span>|</span>
                    <a href="mailto:livia.fernandes@academico.ufpb.br" style="color: #d4a017 !important;">livia.fernandes@academico.ufpb.br</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
