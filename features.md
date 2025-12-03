Os artigos de referência detalham abordagens de extração de atributos, principalmente focadas na análise de Sinais de Falta de Alta Impedância (FAIs). Abaixo estão as principais features extraídas e analisadas nos materiais fornecidos, que podem servir de inspiração e serem citadas no seu trabalho:

1. Métricas Baseadas em Sinais Fundamentais e Harmônicos
A extração de atributos a partir de sinais de tensão e corrente é uma prática central, frequentemente utilizando transformadas de processamento de sinais para analisar o conteúdo harmônico.
• Extração de Harmônicos via Stockwell Transform (ST): O ST (Transformada de Stockwell) é amplamente utilizado, pois combina a referência temporal da Wavelet Transform (WT) com a capacidade de análise de frequência da Fourier Transform (FT).
    ◦ Os estudos extraíram harmônicos da fundamental até a 20ª ordem (e até a 64ª ordem em outra análise).
    ◦ Para o problema de localização de FAIs, os harmônicos mais correlacionados com a distância da falta foram geralmente aqueles entre a 12ª e a 20ª ordem.
    ◦ Para o problema de detecção de FAIs (em um método baseado em limiar), a energia da componente fundamental e a rugosidade da terceira harmônica foram as métricas chave. A rugosidade (do sinal da terceira harmônica) é utilizada para quantificar grandes variações no sinal causadas pela FAI.
• Parâmetros Elétricos Calculados por Harmônico: Após a extração dos harmônicos (usando ST, por exemplo), diversos parâmetros são calculados ciclo a ciclo:
    ◦ Módulo, ângulo e energia da corrente.
    ◦ Módulo, ângulo e energia da tensão.
    ◦ Potência ativa, impedância (tensão dividida pela corrente para cada harmônico) e ângulo do fator de potência.

2. Métricas Estatísticas e de Comparação
As informações extraídas dos sinais harmônicos são então transformadas em métricas estatísticas ao longo de uma janela de tempo (ex: 20 ciclos, 10 pré-falta e 10 pós-falta):
• Estatísticas Comuns: As estatísticas extraídas para compor o dataset incluem média, máximo, desvio padrão (std), variância, mínimo e amplitude (diferença entre máximo e mínimo).
    ◦ Para localização de FAIs, as estatísticas que mostraram a maior correlação (PCC acima de 0,9) com a distância da falta foram: média, máximo, desvio padrão e amplitude dos parâmetros.
• Métricas de Razão (Ratio-based Metrics): Uma descoberta importante em uma das metodologias propostas é que as métricas com maior correlação com a distância da falta (CCP acima de 0,9) eram métricas de razão.
    ◦ Essas métricas são calculadas pela razão entre a métrica obtida em uma medição esparsa (remota, ex: Barramento 858) e a métrica obtida na subestação. Isso é feito para normalizar os dados e amplificar os desvios dependentes da distância, suprimindo efeitos de modos comuns.

3. Sinais de Entrada e Combinações de Atributos Mais Relevantes
Os sinais de entrada e as combinações de atributos que demonstraram melhor desempenho para localização de FAIs foram:
• Sinais de Tensão: Para a localização de FAIs, a energia e o módulo dos harmônicos de tensão (em oposição aos de corrente) demonstraram uma forte correlação com a distância da falta.
• Sinais Agregados (Soma das Três Fases): A análise demonstrou que tanto a soma dos sinais das três fases quanto a medição da fase faltosa eram adequadas, mas a soma das três fases (ou corrente neutra) pode ser vantajosa pois dispensa um algoritmo prévio de classificação de fase faltosa.
• O Conjunto Otimizado (Proposto no Artigo IEEE Access): O método de localização de FAI baseado em Ensemble Tree (ET) usou um conjunto altamente eficaz de 15 métricas:
    ◦ Grandeza Elétrica: Tensão.
    ◦ Sinal: Soma das três fases.
    ◦ Harmônicos: 12ª até 16ª ordem.
    ◦ Parâmetro: Módulo das harmônicas.
    ◦ Estatísticas: Amplitude, Máximo e Desvio Padrão.
    ◦ Consideração: Razão entre a medição esparsa e a subestação.

--------------------------------------------------------------------------------
Citações dos Artigos de Referência para FAI e Localização (Os trabalhos mais detalhados sobre as métricas):
1. Lopes, G. N., Barbalho, P. I. N., Vieira, J. C. M., & Coury, D. V. (2025). High Impedance Fault Location in Distribution Systems: A Novel Approach With Enhanced Metrics and Intelligent Algorithms. IEEE Access, vol. 13, pp. 108466-108480, 2025.
2. Lopes, G. N., Menezes, T. S., Gomes, D. P. S., & Vieira, J. C. M. (2023). High Impedance Fault Location Methods: Review and Harmonic Selection-Based Analysis. IEEE Open Access Journal of Power And Energy, v. 10, pp. 438-449, 2023.
3. Lopes, G. N., Menezes, T. S., & Vieira, J. C. M. (2024). Reliable high impedance fault detection method based on the roughness of the neutral current in active distribution systems. International Journal of Electrical Power and Energy Systems (IJEPES), vol 159.