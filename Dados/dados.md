# Hierarquia dos dados:

- raw/
    - Current_HIF/
        - HF/: Dados de falha (corrente) em alta frequência numerados em .parquet (ex: 10.parquet)
        - LF/: Dados de falha (corrente) em baixa frequência numerados em .parquet (ex: 10.parquet)
    - Currents_cal/: Dados sem falha (corrente) numerados em .parquet (ex: 10.parquet)
    - Voltage_HIF/
        - HF/: Dados de falha (tensao) em alta frequência numerados em .parquet (ex: 10.parquet)
        - LF/: Dados de falha (tensao) em baixa frequência numerados em .parquet (ex: 10.parquet)
    - Voltages_cal/
        - HF/: Dados sem falha (tensao) em alta frequência numerados em .parquet (ex: 10.parquet)
        - LF/: Dados sem falha (tensao) em baixa frequência numerados em .parquet (ex: 10.parquet)
    - Dados_Parquet_zip.rar

Todos os dados contém apenas uma coluna no arquivo parquet, denominada S, que é o sinal.

# Frequências dos dados

| Tipo           | Faixa de frequência | Taxa real     |
| -------------- | ------------------- | ------------- |
| **Voltage LF** | 0–50 kHz            | **100 kSa/s** |
| **Voltage HF** | 10 kHz–1 MHz        | **2 MSa/s**   |
| **Current LF** | 0–50 kHz            | **100 kSa/s** |
| **Current HF** | 10 kHz–1 MHz        | **2 MSa/s**   |

De acordo com o artigo, a frequência da rede é de 50 Hz. Dessa forma, para conseguir até a 20 harmônica: 20x50 = 1000 Hz (1 kHz).
Pelo Teorema da amostragem de Nyquist, a amostragem deve ser de 2x1000 = 2000 Hz = 2 kHz.
Dado que os dados HF cobrem de 10 kHz–1 MHz, optou-se por usar os dados LF.
