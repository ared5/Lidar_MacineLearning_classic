"""Funciones de creación de features extraídas de ML_workflow_newTargets_PRETIME_HubLoads.ipynb.

Archivo regenerado a partir del notebook solicitado por el usuario.
"""

import numpy as np
import pandas as pd


def create_vlos_lags(df, lag_seconds_list=[2, 5, 8, 11, 14, 17, 20, 23, 26]):
    """
    Crea features de lag para las variables de velocidad del viento (VLOS).
    
    Args:
        df: DataFrame con los datos
        lag_seconds_list: Lista de lags en segundos a crear
    
    Returns:
        DataFrame con las nuevas columnas de lag añadidas
    """
    # Identificar columnas de velocidad del viento (VLOS)
    vlos_columns = [col for col in df.columns if 'LAC_VLOS' in col]
    
    print(f"Variables VLOS encontradas: {len(vlos_columns)}")
    for col in vlos_columns:
        print(f"  - {col}")
    
    # Calcular tiempo de muestreo (dt) asumiendo columna Time
    if 'Time' in df.columns:
        dt = df['Time'].iloc[1] - df['Time'].iloc[0]  # Segundos entre muestras
        print(f"\nTiempo de muestreo detectado: {dt:.4f} segundos")
    else:
        dt = 0.02  # Default 50Hz
        print(f"\nTiempo de muestreo por defecto: {dt} segundos")
    
    # Crear lags para cada variable VLOS
    print(f"\nCreando {len(lag_seconds_list)} lags para cada variable VLOS...")
    
    total_created = 0
    for vlos_col in vlos_columns:
        for lag_sec in lag_seconds_list:
            # Calcular numero de muestras para el lag
            lag_samples = int(round(lag_sec / dt))
            
            # Crear nombre de la nueva columna
            new_col_name = f"{vlos_col}_lag{lag_sec}s"
            
            # Crear la columna con shift
            df[new_col_name] = df[vlos_col].shift(lag_samples)
            
            total_created += 1
    
    print(f"Total de features de lag creadas: {total_created}")
    print(f"Shape del DataFrame: {df.shape}")
    
    return df


def create_azimuth_components(df):
    """
    Crea componentes seno y coseno del angulo de azimuth del rotor.
    Esto evita discontinuidades en 0-360 grados.
    
    Args:
        df: DataFrame con los datos
    
    Returns:
        DataFrame con las nuevas columnas sin(azimuth) y cos(azimuth)
    """
    azimuth_col = 'Rotor azimuth angle'
    
    if azimuth_col not in df.columns:
        print(f"ADVERTENCIA: Columna '{azimuth_col}' no encontrada")
        return df
    
    print(f"Creando componentes trigonometricas de '{azimuth_col}'...")
    
    # Crear componentes (asumiendo que el angulo esta en grados)
    # Si esta en radianes, no hace falta convertir
    # Verificar rango de valores para determinar unidades
    max_val = df[azimuth_col].max()
    
    if max_val > 6.5:  # Si es > 2*pi, probablemente en grados
        print(f"   Rango detectado: 0-{max_val:.1f} (grados)")
        # Convertir de grados a radianes
        azimuth_rad = np.deg2rad(df[azimuth_col])
    else:
        print(f"   Rango detectado: 0-{max_val:.1f} (radianes)")
        azimuth_rad = df[azimuth_col]
    
    # Crear componentes
    df['sin_rotor_azimuth'] = np.sin(azimuth_rad)
    df['cos_rotor_azimuth'] = np.cos(azimuth_rad)
    
    print(f"   OK - Creadas 2 nuevas columnas: sin_rotor_azimuth, cos_rotor_azimuth")
    print(f"   Shape del DataFrame: {df.shape}")
    
    return df


def lowpass_filter(signal_data, cutoff, fs, order=2):
    """
    Aplica un filtro pasa-bajo Butterworth a la señal.
    
    Args:
        signal_data (np.array): Señal de entrada
        cutoff (float): Frecuencia de corte en Hz
        fs (float): Frecuencia de muestreo en Hz
        order (int): Orden del filtro
    
    Returns:
        np.array: Señal filtrada
    """
    from scipy import signal as sp_signal
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    # Asegurar que el valor está en el rango válido (0, 1)
    normal_cutoff = max(0.001, min(normal_cutoff, 0.999))
    
    sos = sp_signal.butter(order, normal_cutoff, btype='low', output='sos')
    filtered_signal = sp_signal.sosfilt(sos, signal_data)
    
    return filtered_signal


def lowpass_filter_safe(signal_data, cutoff, fs, order=2):
    """
    Versión segura del filtro pasa-bajo que maneja errores.
    
    Args:
        signal_data (np.array): Señal de entrada
        cutoff (float): Frecuencia de corte en Hz
        fs (float): Frecuencia de muestreo en Hz
        order (int): Orden del filtro
    
    Returns:
        np.array: Señal filtrada, o señal original si falla el filtrado
    """
    try:
        return lowpass_filter(signal_data, cutoff, fs, order)
    except Exception as e:
        print(f"      ADVERTENCIA: Fallo en filtrado pasa-bajo ({e}). Usando señal sin filtrar.")
        return signal_data


def create_frequency_components_1P_2P(df, apply_filtering=True):
    """
    Crea componentes de frecuencia 0P, 1P y 2P de los momentos flectores.
    
    DESCRIPCIÓN:
    Esta función crea los siguientes targets a partir de los momentos flectores M1(t) y M2(t):
    
    1. Señales suma y diferencia:
       - M_Σ(t) = (M1(t) + M2(t)) / 2  → contiene componentes pares (2P, 4P, ...)
       - M_Δ(t) = (M1(t) - M2(t)) / 2  → contiene componentes impares (1P, 3P, ...)
    
    2. Componente 0P (lento/promedio):
       - M_0(t) = M_Σ(t)  → componente lento
    
    3. Componente 1P (proyectado en ejes fijos):
       - M_1c(t) = M_Δ(t) * cos(ψ(t))
       - M_1s(t) = M_Δ(t) * sin(ψ(t))
       donde ψ(t) es el ángulo de azimut de la pala 1
    
    4. Componente 2P (proyectado en ejes fijos):
       - M_2c(t) = M_Σ(t) * cos(2ψ(t))
       - M_2s(t) = M_Σ(t) * sin(2ψ(t))
    
    IMPORTANTE: Para que 1P y 2P sean "limpios", se recomienda filtrar:
       - M_Δ alrededor de 1P antes de proyectar (band-pass)
       - M_Σ alrededor de 2P antes de proyectar (band-pass)
    
    Targets de salida: [M_0, M_1c, M_1s, M_2c, M_2s]
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de la simulación.
                          Debe contener al menos:
                          - 'Time': tiempo en segundos
                          - 'Rotor speed': velocidad del rotor en rpm
                          - 'Rotor azimuth angle': ángulo de azimut del rotor (pala 1)
                          - 'Blade root 1 My': momento flector pala 1
                          - 'Blade root 2 My': momento flector pala 2
        apply_filtering (bool): Si True, aplica filtrado pasa-banda antes de proyectar.
                               Default: True
    
    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas:
                     - 'M_0' (0P): componente lento
                     - 'M_1c' (1P coseno): componente 1P en fase
                     - 'M_1s' (1P seno): componente 1P en cuadratura
                     - 'M_2c' (2P coseno): componente 2P en fase
                     - 'M_2s' (2P seno): componente 2P en cuadratura
    
    Raises:
        ValueError: Si faltan columnas requeridas en el DataFrame.
    """
    
    # Validar columnas requeridas
    required_cols = ['Time', 'Rotor speed', 'Rotor azimuth angle', 
                     'Blade root 1 My', 'Blade root 2 My']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el DataFrame: {missing_cols}")
    
    print("=" * 70)
    print("Creando componentes de frecuencia 0P, 1P y 2P...")
    print("=" * 70)
    
    # =========================================================================
    # PASO 1: Obtener parámetros básicos
    # =========================================================================
    M1 = df['Blade root 1 My'].values
    M2 = df['Blade root 2 My'].values
    time = df['Time'].values
    azimuth = df['Rotor azimuth angle'].values
    rotor_speed_rpm = df['Rotor speed'].values
    
    # Convertir azimut a radianes si está en grados
    if azimuth.max() > 6.5:
        azimuth_rad = np.deg2rad(azimuth)
        print("   Azimut convertido de grados a radianes")
    else:
        azimuth_rad = azimuth
        print("   Azimut ya está en radianes")
    
    # Calcular frecuencias
    freq_1P_Hz = rotor_speed_rpm / 60.0  # Convertir rpm a Hz
    freq_2P_Hz = 2 * freq_1P_Hz
    freq_1P_mean = freq_1P_Hz.mean()
    freq_2P_mean = freq_2P_Hz.mean()
    
    # Calcular frecuencia de muestreo
    if len(df) > 1:
        dt = time[1] - time[0]
        fs = 1.0 / dt
    else:
        dt = 0.02
        fs = 50.0
    
    print(f"\n   Parámetros:")
    print(f"   - Rotor Speed promedio: {rotor_speed_rpm.mean():.2f} rpm")
    print(f"   - Frecuencia 1P promedio: {freq_1P_mean:.3f} Hz")
    print(f"   - Frecuencia 2P promedio: {freq_2P_mean:.3f} Hz")
    print(f"   - Frecuencia de muestreo: {fs:.1f} Hz")
    print(f"   - Número de muestras: {len(df)}")
    
    # =========================================================================
    # PASO 2: Calcular señales suma (Σ) y diferencia (Δ)
    # =========================================================================
    print(f"\n   Calculando M_Σ y M_Δ...")
    
    M_sum = (M1 + M2) / 2.0  # M_Σ: contiene componentes pares (2P, 4P, ...)
    M_diff = (M1 - M2) / 2.0  # M_Δ: contiene componentes impares (1P, 3P, ...)
    
    print(f"   - M_Σ (suma) calculada: rango [{M_sum.min():.2f}, {M_sum.max():.2f}]")
    print(f"   - M_Δ (diferencia) calculada: rango [{M_diff.min():.2f}, {M_diff.max():.2f}]")
    
    # =========================================================================
    # PASO 3: Aplicar filtrado pasa-banda (opcional pero recomendado)
    # =========================================================================
    if apply_filtering:
        print(f"\n   Aplicando filtrado pasa-banda...")
        
        # Filtrar M_Δ alrededor de 1P
        bandwidth_1P = 0.3  # Ancho de banda en Hz alrededor de 1P
        lowcut_1P = max(0.01, freq_1P_mean - bandwidth_1P)
        highcut_1P = min(fs/2 - 0.1, freq_1P_mean + bandwidth_1P)
        
        print(f"   - Filtrando M_Δ alrededor de 1P: [{lowcut_1P:.3f}, {highcut_1P:.3f}] Hz")
        M_diff_filtered = bandpass_filter_safe(M_diff, lowcut_1P, highcut_1P, fs, order=2)
        
        # Filtrar M_Σ alrededor de 2P
        bandwidth_2P = 0.5  # Ancho de banda en Hz alrededor de 2P
        lowcut_2P = max(0.01, freq_2P_mean - bandwidth_2P)
        highcut_2P = min(fs/2 - 0.1, freq_2P_mean + bandwidth_2P)
        
        print(f"   - Filtrando M_Σ alrededor de 2P: [{lowcut_2P:.3f}, {highcut_2P:.3f}] Hz")
        M_sum_filtered = bandpass_filter_safe(M_sum, lowcut_2P, highcut_2P, fs, order=2)
    else:
        print(f"\n   Sin filtrado (apply_filtering=False)")
        M_diff_filtered = M_diff
        M_sum_filtered = M_sum
    
    # =========================================================================
    # PASO 4: Crear componentes 0P, 1P y 2P
    # =========================================================================
    print(f"\n   Creando componentes de frecuencia...")
    
    # 0P: Componente DC (eliminar frecuencias pares 2P, 4P, ...)
    if apply_filtering:
        # Filtro pasa-bajo para quedarse solo con componente DC
        # Corte por debajo de 1P para eliminar 2P, 4P, etc.
        cutoff_0P = freq_1P_mean * 8 # Cortar a la mitad de 1P
        print(f"   - Filtrando M_0 (pasa-bajo) con corte en {cutoff_0P:.3f} Hz")
        M_0 = lowpass_filter_safe(M_sum, cutoff_0P, fs, order=2)
        print(f"   - M_0 (0P): componente DC creado (sin 2P, 4P, ...)")
    else:
        M_0 = M_sum  # Sin filtrar
        print(f"   - M_0 (0P): componente lento creado (sin filtrar)")
    
    # 1P: Proyección de M_Δ en ejes fijos usando azimut
    M_1c = M_diff_filtered * np.cos(azimuth_rad)  # Componente 1P en fase (coseno)
    M_1s = M_diff_filtered * np.sin(azimuth_rad)  # Componente 1P en cuadratura (seno)
    print(f"   - M_1c, M_1s (1P): componentes creadas con proyección en ejes fijos")
    
    # 2P: Proyección de M_Σ en ejes fijos usando 2*azimut
    M_2c = M_sum_filtered * np.cos(2 * azimuth_rad)  # Componente 2P en fase (coseno)
    M_2s = M_sum_filtered * np.sin(2 * azimuth_rad)  # Componente 2P en cuadratura (seno)
    print(f"   - M_2c, M_2s (2P): componentes creadas con proyección en ejes fijos")
    
    # =========================================================================
    # PASO 5: Agregar al DataFrame
    # =========================================================================
    print(f"\n   Agregando columnas al DataFrame...")
    
    df['M_0'] = M_0      # 0P
    df['M_1c'] = M_1c    # 1P coseno
    df['M_1s'] = M_1s    # 1P seno
    df['M_2c'] = M_2c    # 2P coseno
    df['M_2s'] = M_2s    # 2P seno
    
    new_columns = ['M_0', 'M_1c', 'M_1s', 'M_2c', 'M_2s']
    print(f"   - Columnas creadas: {new_columns}")
    
    # =========================================================================
    # PASO 6: Resumen final
    # =========================================================================
    print(f"\n" + "=" * 70)
    print(f"RESUMEN:")
    print(f"=" * 70)
    print(f"   Vector de salida: y(t) = [M_0, M_1c, M_1s, M_2c, M_2s]")
    print(f"   - M_0:  componente 0P (lento)")
    print(f"   - M_1c: componente 1P en fase (coseno)")
    print(f"   - M_1s: componente 1P en cuadratura (seno)")
    print(f"   - M_2c: componente 2P en fase (coseno)")
    print(f"   - M_2s: componente 2P en cuadratura (seno)")
    print(f"\n   Shape final del DataFrame: {df.shape}")
    print(f"=" * 70)
    
    return df


def bandpass_filter(signal_data, lowcut, highcut, fs, order=2):
    """
    Aplica un filtro pasa-banda Butterworth a la señal.
    
    Args:
        signal_data (np.array): Señal de entrada
        lowcut (float): Frecuencia de corte inferior en Hz
        highcut (float): Frecuencia de corte superior en Hz
        fs (float): Frecuencia de muestreo en Hz
        order (int): Orden del filtro
    
    Returns:
        np.array: Señal filtrada
    """
    from scipy import signal as sp_signal
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Asegurar que los valores están en el rango válido (0, 1)
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    sos = sp_signal.butter(order, [low, high], btype='band', output='sos')
    filtered_signal = sp_signal.sosfilt(sos, signal_data)
    
    return filtered_signal


def bandpass_filter_safe(signal_data, lowcut, highcut, fs, order=2):
    """
    Versión segura del filtro pasa-banda que maneja errores.
    
    Args:
        signal_data (np.array): Señal de entrada
        lowcut (float): Frecuencia de corte inferior en Hz
        highcut (float): Frecuencia de corte superior en Hz
        fs (float): Frecuencia de muestreo en Hz
        order (int): Orden del filtro
    
    Returns:
        np.array: Señal filtrada, o señal original si falla el filtrado
    """
    try:
        return bandpass_filter(signal_data, lowcut, highcut, fs, order)
    except Exception as e:
        print(f"      ADVERTENCIA: Fallo en filtrado ({e}). Usando señal sin filtrar.")
        return signal_data


def create_pitch_coleman_features(df):
    """
    Crea features de pitch en marco coherente (Coleman transformation).
    
    DESCRIPCIÓN:
    Esta función transforma los ángulos de pitch de las palas individuales a un marco
    de referencia fijo (no rotatorio) usando la transformación de Coleman:
    
    1. Componentes colectivo y diferencial:
       - θ_0(t) = (θ_1(t) + θ_2(t)) / 2  → colectivo (promedio)
       - θ_Δ(t) = (θ_1(t) - θ_2(t)) / 2  → diferencial
    
    2. Proyección del diferencial a ejes fijos (1P):
       - θ_1c(t) = θ_Δ(t) * cos(ψ(t))
       - θ_1s(t) = θ_Δ(t) * sin(ψ(t))
       donde ψ(t) es el ángulo de azimut
    
    3. Rates (derivadas temporales):
       - θ̇_0(t) ≈ (θ_0(t) - θ_0(t-Δt)) / Δt
       - θ̇_1c(t), θ̇_1s(t) de forma similar
    
    4. Rotor speed rate:
       - Ω̇(t) ≈ (Ω(t) - Ω(t-Δt)) / Δt
    
    IMPORTANTE: Estos features son coherentes con targets en Coleman (M_0, M_1c, M_1s).
    
    Args:
        df (pd.DataFrame): DataFrame con los datos. Debe contener:
                          - 'Time': tiempo en segundos
                          - 'Blade 1 pitch angle': pitch pala 1 (grados)
                          - 'Blade 2 pitch angle': pitch pala 2 (grados)
                          - 'Rotor azimuth angle': ángulo de azimut
                          - 'Rotor speed': velocidad del rotor (rpm)
    
    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas:
                     - 'pitch_0': componente colectivo θ_0
                     - 'pitch_1c': componente 1P coseno θ_1c
                     - 'pitch_1s': componente 1P seno θ_1s
                     - 'pitch_0_rate': derivada temporal de θ_0
                     - 'pitch_1c_rate': derivada temporal de θ_1c
                     - 'pitch_1s_rate': derivada temporal de θ_1s
                     - 'rotor_speed_rate': derivada temporal de Ω
    
    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    
    # Validar columnas requeridas
    required_cols = ['Time', 'Blade 1 pitch angle', 'Blade 2 pitch angle', 
                     'Rotor azimuth angle', 'Rotor speed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Columnas faltantes para pitch Coleman: {missing_cols}")
    
    print("=" * 70)
    print("Creando features de pitch en marco Coleman...")
    print("=" * 70)
    
    # =========================================================================
    # PASO 1: Obtener datos básicos
    # =========================================================================
    theta_1 = df['Blade 1 pitch angle'].values
    theta_2 = df['Blade 2 pitch angle'].values
    time = df['Time'].values
    azimuth = df['Rotor azimuth angle'].values
    rotor_speed = df['Rotor speed'].values
    
    # Convertir azimut a radianes si está en grados
    if azimuth.max() > 6.5:
        azimuth_rad = np.deg2rad(azimuth)
        print("   Azimut convertido de grados a radianes")
    else:
        azimuth_rad = azimuth
        print("   Azimut ya está en radianes")
    
    # Calcular dt (paso de tiempo)
    if len(time) > 1:
        dt = time[1] - time[0]
    else:
        dt = 0.1  # default
    
    print(f"\n   Parámetros:")
    print(f"   - Número de muestras: {len(df)}")
    print(f"   - Δt (paso temporal): {dt:.4f} s")
    print(f"   - Frecuencia de muestreo: {1.0/dt:.1f} Hz")
    
    # =========================================================================
    # PASO 2: Transformación Coleman - Colectivo y Diferencial
    # =========================================================================
    print(f"\n   [1/3] Calculando componentes colectivo y diferencial...")
    
    # θ_0: componente colectivo (promedio)
    theta_0 = (theta_1 + theta_2) / 2.0
    
    # θ_Δ: componente diferencial
    theta_delta = (theta_1 - theta_2) / 2.0
    
    print(f"   - θ_0 (colectivo): rango [{theta_0.min():.2f}, {theta_0.max():.2f}]°")
    print(f"   - θ_Δ (diferencial): rango [{theta_delta.min():.2f}, {theta_delta.max():.2f}]°")
    
    # =========================================================================
    # PASO 3: Proyección del diferencial a ejes fijos (1P)
    # =========================================================================
    print(f"\n   [2/3] Proyectando θ_Δ a ejes fijos (1P)...")
    
    # θ_1c: componente 1P en fase (coseno)
    theta_1c = theta_delta * np.cos(azimuth_rad)
    
    # θ_1s: componente 1P en cuadratura (seno)
    theta_1s = theta_delta * np.sin(azimuth_rad)
    
    print(f"   - θ_1c (1P coseno): rango [{theta_1c.min():.2f}, {theta_1c.max():.2f}]°")
    print(f"   - θ_1s (1P seno): rango [{theta_1s.min():.2f}, {theta_1s.max():.2f}]°")
    
    # =========================================================================
    # PASO 4: Calcular rates (derivadas temporales)
    # =========================================================================
    print(f"\n   [3/3] Calculando rates (derivadas temporales)...")
    
    # Derivadas usando diferencias finitas hacia atrás
    # rate(t) ≈ (value(t) - value(t-Δt)) / Δt
    
    # θ̇_0: rate del colectivo
    theta_0_rate = np.zeros_like(theta_0)
    theta_0_rate[1:] = (theta_0[1:] - theta_0[:-1]) / dt
    theta_0_rate[0] = theta_0_rate[1]  # primera muestra = segunda
    
    # θ̇_1c: rate de 1P coseno
    theta_1c_rate = np.zeros_like(theta_1c)
    theta_1c_rate[1:] = (theta_1c[1:] - theta_1c[:-1]) / dt
    theta_1c_rate[0] = theta_1c_rate[1]
    
    # θ̇_1s: rate de 1P seno
    theta_1s_rate = np.zeros_like(theta_1s)
    theta_1s_rate[1:] = (theta_1s[1:] - theta_1s[:-1]) / dt
    theta_1s_rate[0] = theta_1s_rate[1]
    
    # Ω̇: rate de rotor speed
    rotor_speed_rate = np.zeros_like(rotor_speed)
    rotor_speed_rate[1:] = (rotor_speed[1:] - rotor_speed[:-1]) / dt
    rotor_speed_rate[0] = rotor_speed_rate[1]
    
    print(f"   - θ̇_0 rate: rango [{theta_0_rate.min():.2f}, {theta_0_rate.max():.2f}] °/s")
    print(f"   - θ̇_1c rate: rango [{theta_1c_rate.min():.2f}, {theta_1c_rate.max():.2f}] °/s")
    print(f"   - θ̇_1s rate: rango [{theta_1s_rate.min():.2f}, {theta_1s_rate.max():.2f}] °/s")
    print(f"   - Ω̇ rate: rango [{rotor_speed_rate.min():.2f}, {rotor_speed_rate.max():.2f}] rpm/s")
    
    # =========================================================================
    # PASO 5: Agregar al DataFrame
    # =========================================================================
    print(f"\n   Agregando columnas al DataFrame...")
    
    df['pitch_0'] = theta_0
    df['pitch_1c'] = theta_1c
    df['pitch_1s'] = theta_1s
    df['pitch_0_rate'] = theta_0_rate
    df['pitch_1c_rate'] = theta_1c_rate
    df['pitch_1s_rate'] = theta_1s_rate
    df['rotor_speed_rate'] = rotor_speed_rate
    
    new_columns = ['pitch_0', 'pitch_1c', 'pitch_1s', 
                   'pitch_0_rate', 'pitch_1c_rate', 'pitch_1s_rate',
                   'rotor_speed_rate']
    
    print(f"   - Columnas creadas: {len(new_columns)}")
    
    # =========================================================================
    # PASO 6: Resumen final
    # =========================================================================
    print(f"\n" + "=" * 70)
    print(f"RESUMEN:")
    print(f"=" * 70)
    print(f"   Features Coleman de pitch creados:")
    print(f"   - pitch_0:  colectivo θ_0 = (θ_1 + θ_2)/2")
    print(f"   - pitch_1c: 1P coseno θ_1c = θ_Δ·cos(ψ)")
    print(f"   - pitch_1s: 1P seno θ_1s = θ_Δ·sin(ψ)")
    print(f"\n   Rates (derivadas temporales):")
    print(f"   - pitch_0_rate:  θ̇_0")
    print(f"   - pitch_1c_rate: θ̇_1c")
    print(f"   - pitch_1s_rate: θ̇_1s")
    print(f"   - rotor_speed_rate: Ω̇")
    print(f"\n   💡 Estos features son coherentes con targets Coleman (M_0, M_1c, M_1s)")
    print(f"   💡 Los rates capturan dinámica → mejoran predicción de componentes 1P")
    print(f"\n   Shape final del DataFrame: {df.shape}")
    print(f"=" * 70)
    
    return df


def create_wind_field_statistics(df, rotation_offset=None):
    """
    Crea estadísticas del campo de viento LIDAR con configuración ROTABLE de beams.
    
    PARÁMETROS DE ROTACIÓN:
    - rotation_offset: Número de posiciones a rotar (None usa ROTATION_OFFSET global)
                      +1 = rotar 45° CW, -1 = rotar 45° CCW
    
    DESCRIPCIÓN:
    Esta función calcula características agregadas del campo de viento medido por el LIDAR
    que capturan:
    1. Intensidad del viento (media)
    2. Turbulencia/heterogeneidad (desviación estándar)
    3. Shear vertical (gradiente arriba-abajo)
    4. Gradiente horizontal (diferencia izquierda-derecha, relacionado con yaw misalignment)
    
    FÓRMULAS:
    - U_mean = mean(VLOS de todos los BEAMs válidos)  → ayuda a predecir M_0
    - U_std = std(VLOS de todos los BEAMs válidos)    → captura turbulencia/heterogeneidad
    - U_shear_vert = mean(BEAMs arriba) - mean(BEAMs abajo)  → shear vertical
    - U_shear_horiz = mean(BEAMs izquierda) - mean(BEAMs derecha)  → gradiente lateral
    
    CONFIGURACIÓN BASE (antes de rotar):
                    0° (↑)
                    BEAM 0
                     |
        315° BEAM 7  |  45° BEAM 1
               ╲     |     ╱
                ╲    |    ╱
        270° ────────+──────── 90°
        BEAM 6       |       BEAM 2
                ╱    |    ╲
               ╱     |     ╲
        225° BEAM 5  |  135° BEAM 3
                     |
                  BEAM 4
                 180° (↓)
    
    Args:
        df (pd.DataFrame): DataFrame con columnas LAC_VLOS de diferentes BEAMs.
        rotation_offset (int): Offset de rotación (None usa valor global)
    
    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas:
                     - 'U_mean': velocidad media del campo de viento
                     - 'U_std': desviación estándar (heterogeneidad)
                     - 'U_shear_vert': shear vertical (arriba - abajo)
                     - 'U_shear_horiz': gradiente horizontal (izquierda - derecha)
    
    Raises:
        ValueError: Si no se encuentran columnas VLOS en el DataFrame.
    """
    
    # Usar valor global si no se especifica
    if rotation_offset is None:
        rotation_offset = ROTATION_OFFSET
    
    print("=" * 70)
    print("Creando estadísticas del campo de viento LIDAR...")
    print("=" * 70)
    print(f"🔄 ROTACIÓN APLICADA: {rotation_offset} posiciones ({rotation_offset * 45}°)")
    
    # =========================================================================
    # PASO 1: Identificar columnas VLOS (sin lag) y FILTRAR BEAMS VACÍOS
    # =========================================================================
    print("\n   [1/4] Identificando y filtrando columnas VLOS...")
    
    # Buscar todas las columnas que contengan 'LAC_VLOS' pero NO 'lag'
    vlos_cols = [col for col in df.columns if 'LAC_VLOS' in col and 'lag' not in col.lower()]
    
    if len(vlos_cols) == 0:
        raise ValueError("No se encontraron columnas LAC_VLOS en el DataFrame")
    
    print(f"   - Columnas VLOS encontradas: {len(vlos_cols)}")
    
    # Extraer números de BEAM y FILTRAR por porcentaje de NaNs
    beam_numbers = []
    beam_to_col = {}
    
    for col in vlos_cols:
        match = re.search(r'BEAM(\d+)', col)
        if match:
            beam_num = int(match.group(1))
            
            # FILTRADO: Calcular porcentaje de NaNs
            nan_percentage = df[col].isna().sum() / len(df) * 100
            
            if nan_percentage > 90:
                # Beam vacío → ignorar
                print(f"   ⚠️  BEAM {beam_num} ignorado ({nan_percentage:.1f}% NaNs)")
                continue
            else:
                # Beam válido → incluir
                beam_numbers.append(beam_num)
                beam_to_col[beam_num] = col
                print(f"   ✓  BEAM {beam_num} válido ({nan_percentage:.1f}% NaNs)")
    
    beam_numbers = sorted(set(beam_numbers))
    print(f"\n   → BEAMs válidos detectados: {beam_numbers}")
    
    # =========================================================================
    # PASO 2: Configuración de posiciones de BEAMs CON ROTACIÓN
    # =========================================================================
    print("\n   [2/4] Configurando posiciones de BEAMs...")
    
    num_beams = len(beam_numbers)
    
    # CONFIGURACIÓN ESPECÍFICA PARA 8 BEAMS (0-7) CON ROTACIÓN
    if num_beams == 8:
        print(f"   → Configuración de 8 BEAMs (distribuidos cada 45°):")
        
        # CONFIGURACIÓN BASE (sin rotar)
        # Arriba: 0°, 45°, 315° → beams [0, 1, 7]
        # Abajo: 135°, 180°, 225° → beams [3, 4, 5]
        # Izquierda: 225°, 270°, 315° → beams [5, 6, 7]
        # Derecha: 45°, 90°, 135° → beams [1, 2, 3]
        
        # APLICAR ROTACIÓN: cada beam se mueve rotation_offset posiciones
        def rotate_beam(beam_num, offset):
            """Rota un beam aplicando módulo 8"""
            return (beam_num + offset) % 8
        
        # Rotar cada grupo
        beams_up_base = [0, 1, 7]
        beams_down_base = [3, 4, 5]
        beams_left_base = [5, 6, 7]
        beams_right_base = [1, 2, 3]
        
        beams_up = [rotate_beam(b, rotation_offset) for b in beams_up_base]
        beams_down = [rotate_beam(b, rotation_offset) for b in beams_down_base]
        beams_left = [rotate_beam(b, rotation_offset) for b in beams_left_base]
        beams_right = [rotate_beam(b, rotation_offset) for b in beams_right_base]
        
        # Calcular ángulos correspondientes
        angles_up = [(b * 45) % 360 for b in beams_up]
        angles_down = [(b * 45) % 360 for b in beams_down]
        angles_left = [(b * 45) % 360 for b in beams_left]
        angles_right = [(b * 45) % 360 for b in beams_right]
        
        print(f"      Arriba (↑):    {beams_up}    → ángulos: {angles_up}")
        print(f"      Abajo (↓):     {beams_down}    → ángulos: {angles_down}")
        print(f"      Izquierda (←): {beams_left}    → ángulos: {angles_left}")
        print(f"      Derecha (→):   {beams_right}    → ángulos: {angles_right}")
        
    else:
        # CONFIGURACIÓN GENÉRICA para otros números de beams
        print(f"   → Configuración genérica para {num_beams} BEAMs:")
        
        if num_beams >= 4:
            # Dividir en cuadrantes aproximadamente
            quarter = num_beams // 4
            
            beams_up = beam_numbers[:quarter + 1]
            beams_right = beam_numbers[quarter:2*quarter + 1]
            beams_down = beam_numbers[2*quarter:3*quarter + 1]
            beams_left = beam_numbers[3*quarter:] + beam_numbers[:1]
            
        else:
            # Si hay muy pocos BEAMs, usar todos para cada cálculo
            print("   ⚠️  Pocos BEAMs detectados. Usando configuración simplificada.")
            beams_up = beam_numbers[:len(beam_numbers)//2]
            beams_down = beam_numbers[len(beam_numbers)//2:]
            beams_left = beam_numbers[:len(beam_numbers)//2]
            beams_right = beam_numbers[len(beam_numbers)//2:]
        
        print(f"      Arriba (↑):    {beams_up}")
        print(f"      Derecha (→):   {beams_right}")
        print(f"      Abajo (↓):     {beams_down}")
        print(f"      Izquierda (←): {beams_left}")
    
    # =========================================================================
    # PASO 3: Calcular estadísticas del campo de viento
    # =========================================================================
    print("\n   [3/4] Calculando estadísticas...")
    
    # Crear DataFrame solo con columnas VLOS válidas para cálculos eficientes
    vlos_data = df[[beam_to_col[b] for b in beam_numbers]]
    
    # --- 3.1: U_mean (media del campo de viento) ---
    U_mean = vlos_data.mean(axis=1)
    print(f"   - U_mean: rango [{U_mean.min():.2f}, {U_mean.max():.2f}] m/s")
    
    # --- 3.2: U_std (heterogeneidad/turbulencia) ---
    U_std = vlos_data.std(axis=1)
    print(f"   - U_std: rango [{U_std.min():.2f}, {U_std.max():.2f}] m/s")
    
    # --- 3.3: U_shear_vert (shear vertical: arriba - abajo) ---
    if len(beams_up) > 0 and len(beams_down) > 0:
        cols_up = [beam_to_col[b] for b in beams_up if b in beam_to_col]
        cols_down = [beam_to_col[b] for b in beams_down if b in beam_to_col]
        
        U_up = df[cols_up].mean(axis=1)
        U_down = df[cols_down].mean(axis=1)
        U_shear_vert = U_up - U_down
        
        print(f"   - U_shear_vert: rango [{U_shear_vert.min():.2f}, {U_shear_vert.max():.2f}] m/s")
        print(f"                   media: {U_shear_vert.mean():.3f} m/s")
    else:
        U_shear_vert = pd.Series(0.0, index=df.index)
        print(f"   - U_shear_vert: no se pudo calcular (BEAMs insuficientes)")
    
    # --- 3.4: U_shear_horiz (gradiente horizontal: izquierda - derecha) ---
    if len(beams_left) > 0 and len(beams_right) > 0:
        cols_left = [beam_to_col[b] for b in beams_left if b in beam_to_col]
        cols_right = [beam_to_col[b] for b in beams_right if b in beam_to_col]
        
        U_left = df[cols_left].mean(axis=1)
        U_right = df[cols_right].mean(axis=1)
        U_shear_horiz = U_left - U_right
        
        print(f"   - U_shear_horiz: rango [{U_shear_horiz.min():.2f}, {U_shear_horiz.max():.2f}] m/s")
        print(f"                    media: {U_shear_horiz.mean():.3f} m/s")
    else:
        U_shear_horiz = pd.Series(0.0, index=df.index)
        print(f"   - U_shear_horiz: no se pudo calcular (BEAMs insuficientes)")
    
    # =========================================================================
    # PASO 4: Agregar al DataFrame
    # =========================================================================
    print("\n   [4/4] Agregando columnas al DataFrame...")
    
    df['U_mean'] = U_mean
    df['U_std'] = U_std
    df['U_shear_vert'] = U_shear_vert
    df['U_shear_horiz'] = U_shear_horiz
    
    new_columns = ['U_mean', 'U_std', 'U_shear_vert', 'U_shear_horiz']
    
    # =========================================================================
    # PASO 5: Resumen final
    # =========================================================================
    print(f"\n" + "=" * 70)
    print(f"RESUMEN:")
    print(f"=" * 70)
    print(f"   Estadísticas del campo de viento creadas:")
    print(f"   - U_mean:        velocidad media → predice M_0")
    print(f"   - U_std:         heterogeneidad/turbulencia")
    print(f"   - U_shear_vert:  shear vertical (↑ - ↓)")
    print(f"   - U_shear_horiz: gradiente lateral (← - →)")
    print(f"\n   💡 Estas variables capturan la estructura espacial del viento")
    print(f"   💡 U_shear_vert y U_shear_horiz ayudan a predecir componentes 1P")
    print(f"   💡 BEAMs vacíos (>90% NaN) fueron filtrados automáticamente")
    print(f"   🔄 Rotación aplicada: {rotation_offset} × 45° = {rotation_offset * 45}°")
    print(f"\n   Shape final del DataFrame: {df.shape}")
    print(f"=" * 70)
    
    return df


def create_wind_statistics_lags(df, lag_times=[2, 5, 8, 11, 14, 17, 20, 23, 26]):
    """
    Crea lags de las estadísticas del campo de viento (U_mean, U_std, U_shear_vert, U_shear_horiz).
    
    DESCRIPCIÓN:
    Esta función crea versiones desplazadas temporalmente de las estadísticas del viento,
    permitiendo al modelo capturar cómo las condiciones de viento pasadas afectan las
    cargas actuales en las palas.
    
    IMPORTANTE: Ejecutar después de create_wind_field_statistics().
    
    Args:
        df (pd.DataFrame): DataFrame con las columnas U_mean, U_std, U_shear_vert, U_shear_horiz
        lag_times (list): Lista de tiempos de lag en segundos
    
    Returns:
        pd.DataFrame: DataFrame con columnas adicionales:
                     - 'U_mean_lag{X}s', 'U_std_lag{X}s', etc. para cada lag
    
    Raises:
        ValueError: Si faltan columnas de estadísticas de viento.
    """
    
    # Validar que existen las columnas base
    required_cols = ['U_mean', 'U_std', 'U_shear_vert', 'U_shear_horiz', 'Time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Faltan columnas de estadísticas de viento: {missing_cols}. "
                        f"Ejecuta create_wind_field_statistics() primero.")
    
    print("=" * 70)
    print("Creando lags de estadísticas del campo de viento...")
    print("=" * 70)
    
    # Calcular sampling rate
    time_values = df['Time'].values
    if len(time_values) > 1:
        dt = time_values[1] - time_values[0]
        fs = 1.0 / dt
    else:
        dt = 0.1
        fs = 10.0
    
    print(f"\n   Parámetros:")
    print(f"   - Sampling rate: {fs:.1f} Hz (dt = {dt:.3f} s)")
    print(f"   - Lags a crear: {lag_times[0]}s - {lag_times[-1]}s ({len(lag_times)} lags)")
    
    # Variables base
    base_vars = ['U_mean', 'U_std', 'U_shear_vert', 'U_shear_horiz']
    
    print(f"   - Variables base: {len(base_vars)}")
    
    # Contador de columnas creadas
    created_cols = 0
    
    # Crear lags para cada variable
    for var in base_vars:
        for lag_sec in lag_times:
            # Calcular lag en muestras
            lag_samples = int(round(lag_sec * fs))
            
            # Nombre de la nueva columna
            col_name = f"{var}_lag{lag_sec}s"
            
            # Crear lag usando shift
            df[col_name] = df[var].shift(lag_samples)
            
            created_cols += 1
    
    print(f"\n   ✅ Columnas de lag creadas: {created_cols}")
    print(f"      ({len(base_vars)} variables × {len(lag_times)} lags)")
    
    # Resumen
    print(f"\n" + "=" * 70)
    print(f"RESUMEN:")
    print(f"=" * 70)
    print(f"   Lags de estadísticas de viento creados:")
    print(f"   - U_mean_lag{lag_times[0]}s ... U_mean_lag{lag_times[-1]}s")
    print(f"   - U_std_lag{lag_times[0]}s ... U_std_lag{lag_times[-1]}s")
    print(f"   - U_shear_vert_lag{lag_times[0]}s ... U_shear_vert_lag{lag_times[-1]}s")
    print(f"   - U_shear_horiz_lag{lag_times[0]}s ... U_shear_horiz_lag{lag_times[-1]}s")
    print(f"\n   💡 Total de nuevas features: {created_cols}")
    print(f"   💡 Estas capturas temporales del viento son cruciales para la predicción")
    print(f"\n   Shape final del DataFrame: {df.shape}")
    print(f"=" * 70)
    
    return df
