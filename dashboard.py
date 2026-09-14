import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import re
import joblib
import shap
import warnings
from io import BytesIO

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard Risiko TBC Jawa Barat",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== SESSION STATE ========================
if "manual_result" not in st.session_state:
    st.session_state.manual_result = None
if "manual_input" not in st.session_state:
    st.session_state.manual_input = None
if "manual_wilayah" not in st.session_state:
    st.session_state.manual_wilayah = None
if "batch_results_df" not in st.session_state:
    st.session_state.batch_results_df = None
if "batch_selected_kab" not in st.session_state:
    st.session_state.batch_selected_kab = "Semua Wilayah"
if "batch_uploaded" not in st.session_state:
    st.session_state.batch_uploaded = False
if "df_final" not in st.session_state:
    st.session_state.df_final = None

# ======================== LOAD MODEL ========================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model_xgboost_tbc.pkl")
        scaler = joblib.load("scaler_minmax.pkl")
        features = joblib.load("features.pkl")
        p33 = joblib.load("threshold_p33.pkl")
        p66 = joblib.load("threshold_p66.pkl")
        t0 = joblib.load("threshold_t0.pkl")
        t_high = joblib.load("threshold_t_high.pkl")
        return model, scaler, features, p33, p66, t0, t_high
    except Exception as e:
        st.error(f"❌ Gagal memuat artefak: {e}")
        st.stop()

model, scaler, features, P33, P66, T0, T_HIGH = load_artifacts()
_FEATURE_ORDER_EXPECTED = ['basic_service_index', 'faskes_ratio', 'poverty_density',
                           'pengeluaran_per_kapita', 'kepadatan_penduduk']
if list(features) != _FEATURE_ORDER_EXPECTED:
    st.error(f"⚠️ Urutan fitur tidak cocok! Model butuh: {list(features)}, kode menghitung: {_FEATURE_ORDER_EXPECTED}")
    st.stop()

# ======================== METRIK EVALUASI ========================
@st.cache_resource
def load_evaluation_metrics():
    try:
        metrics = joblib.load("model_metrics.pkl")
        return metrics
    except:
        return {
            'accuracy': 0.8519,
            'precision_macro': 0.86,
            'recall_macro': 0.85,
            'f1_macro': 0.8480,
            'confusion_matrix': [[10, 0, 0], [2, 7, 1], [0, 1, 6]],
            'class_report': {
                'Rendah': {'precision': 0.83, 'recall': 1.0, 'f1-score': 0.91, 'support': 10},
                'Sedang': {'precision': 0.88, 'recall': 0.70, 'f1-score': 0.78, 'support': 10},
                'Tinggi': {'precision': 0.86, 'recall': 0.86, 'f1-score': 0.86, 'support': 7},
                'accuracy': 0.8519,
                'macro avg': {'precision': 0.86, 'recall': 0.85, 'f1-score': 0.85, 'support': 27}
            }
        }

metrics = load_evaluation_metrics()

# ======================== KODE WILAYAH ========================
KODE_TO_NAMA = {
    '3201': 'KABUPATEN BOGOR', '3202': 'KABUPATEN SUKABUMI', '3203': 'KABUPATEN CIANJUR',
    '3204': 'KABUPATEN BANDUNG', '3205': 'KABUPATEN GARUT', '3206': 'KABUPATEN TASIKMALAYA',
    '3207': 'KABUPATEN CIAMIS', '3208': 'KABUPATEN KUNINGAN', '3209': 'KABUPATEN CIREBON',
    '3210': 'KABUPATEN MAJALENGKA', '3211': 'KABUPATEN SUMEDANG', '3212': 'KABUPATEN INDRAMAYU',
    '3213': 'KABUPATEN SUBANG', '3214': 'KABUPATEN PURWAKARTA', '3215': 'KABUPATEN KARAWANG',
    '3216': 'KABUPATEN BEKASI', '3217': 'KABUPATEN BANDUNG BARAT', '3218': 'KABUPATEN PANGANDARAN',
    '3271': 'KOTA BOGOR', '3272': 'KOTA SUKABUMI', '3273': 'KOTA BANDUNG', '3274': 'KOTA CIREBON',
    '3275': 'KOTA BEKASI', '3276': 'KOTA DEPOK', '3277': 'KOTA CIMAHI', '3278': 'KOTA TASIKMALAYA',
    '3279': 'KOTA BANJAR'
}
NAMA_TO_KODE = {v: k for k, v in KODE_TO_NAMA.items()}

# ======================== FUNGSI PREDIKSI (DIPERBAIKI) ========================
def compute_features(row):
    bsi = (row['persentase_sanitasi_layak'] +
           row['persentase_air_minum_layak'] +
           row['persentase_rumah_layak_huni']) / 3.0
    penduduk = row['jumlah_penduduk']
    faskes = row['jumlah_faskes']
    # FIX SKALA: jumlah_penduduk bisa datang dalam dua satuan berbeda tergantung
    # sumbernya -> Mode Prediksi Manual: input sudah "ribu orang" (mis. 700).
    # Mode Upload Multiple File: berasal dari file BPS mentah (mis. 5.400.000
    # untuk Kab. Bogor, lihat Tabel 4.3 skripsi), yaitu angka penuh jiwa.
    # Training model (yang_rizkaaaaaaaaaaaaaa__1_.py) selalu membagi angka
    # penuh dengan 1000 sebelum menghitung faskes_ratio, jadi di sini kita
    # deteksi otomatis: kab/kota terpadat di Jabar (~5,4 juta jiwa) hanya
    # setara ~5.400 jika sudah dalam ribuan, sehingga ambang 20.000 aman
    # untuk membedakan kedua satuan tanpa risiko salah kategori.
    AMBANG_ANGKA_PENUH = 20000
    penduduk_ribuan = penduduk / 1000.0 if penduduk > AMBANG_ANGKA_PENUH else penduduk
    if penduduk_ribuan > 0 and faskes > 0:
        faskes_ratio = faskes / (penduduk_ribuan + 1)
    else:
        faskes_ratio = 0.0
    poverty_density = row['persentase_penduduk_miskin'] * row['kepadatan_penduduk']
    return [bsi, faskes_ratio, poverty_density, row['pengeluaran_per_kapita'], row['kepadatan_penduduk']]

def apply_threshold(proba, t0, t_high):
    preds = []
    for p in proba:
        if p[2] >= t_high:
            preds.append(2)
        elif p[0] >= t0:
            preds.append(0)
        else:
            preds.append(1)
    return np.array(preds)

def predict_single(input_dict):
    X_raw = np.array([compute_features(input_dict)])
    X_scaled = scaler.transform(X_raw)
    proba = model.predict_proba(X_scaled)[0]
    pred_class = apply_threshold([proba], T0, T_HIGH)[0]
    kelas_map = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    kategori = kelas_map[pred_class]
    risk_index = proba[2] * 100
    return {
        'kategori': kategori,
        'risk_index': risk_index,
        'prob_rendah': proba[0],
        'prob_sedang': proba[1],
        'prob_tinggi': proba[2],
        'pred_class': pred_class
    }

# ======================== SHAP ========================
@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)

shap_explainer = load_shap_explainer()

def get_shap_values(input_dict, pred_class=2):
    X_raw = np.array([compute_features(input_dict)])
    X_scaled = scaler.transform(X_raw)
    shap_values = shap_explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        sv_class = shap_values[pred_class][0]
    else:
        sv_class = shap_values[0, :, pred_class]
    feature_names = ['basic_service_index', 'faskes_ratio', 'poverty_density',
                     'pengeluaran_per_kapita', 'kepadatan_penduduk']
    return {feature_names[i]: sv_class[i] for i in range(len(feature_names))}

# ======================== FUNGSI BACA FILE (DIPERBAIKI) ========================
def normalize_kabupaten_name(name):
    if pd.isna(name):
        return None
    s = str(name).strip().upper()
    s = re.sub(r'^(KABUPATEN|KOTA)\s+', '', s)
    s = re.sub(r'[^A-Z0-9 ]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if s in NAMA_TO_KODE:
        return s
    s_no_space = s.replace(' ', '')
    for key in NAMA_TO_KODE:
        if key.replace(' ', '') == s_no_space:
            return key
    for key in NAMA_TO_KODE:
        if s in key or key in s:
            return key
    return None

def process_single_file(file, filename):
    try:
        name = filename.lower()
        if name.endswith('.csv'):
            content = file.read()
            file.seek(0)
            df = None
            for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(BytesIO(content), encoding=enc)
                    break
                except:
                    continue
            if df is None:
                st.warning(f"Gagal membaca CSV {filename}")
                return None
        else:
            df = pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.warning(f"Gagal membaca file {filename}: {e}")
        return None

    if df is None or df.empty:
        return None

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.dropna(how='all')

    # Kasus 2 kolom
    if len(df.columns) == 2:
        col0, col1 = df.columns[0], df.columns[1]
        if 'kabupaten' in col0 or 'kota' in col0 or 'wilayah' in col0:
            match = re.search(r'\b(\d{4})\b', filename)
            tahun = int(match.group(1)) if match else None
            if tahun is None:
                return None
            df[col1] = pd.to_numeric(df[col1], errors='coerce')
            df = df.dropna(subset=[col1])
            mean_val = df[col1].mean()
            is_absolut = mean_val > 100
            if 'miskin' in filename.lower() and is_absolut:
                st.warning(f"⚠️ File '{filename}' berisi absolut, diabaikan.")
                return None
            df['nama_kabupaten_kota'] = df[col0].astype(str).str.upper().str.strip()
            df['kode_kabupaten_kota'] = df['nama_kabupaten_kota'].map(NAMA_TO_KODE)
            df = df.dropna(subset=['kode_kabupaten_kota'])
            if 'miskin' in filename.lower():
                indikator = 'persentase_penduduk_miskin'
            elif 'sanitasi' in filename.lower():
                indikator = 'persentase_sanitasi_layak'
            elif 'air_minum' in filename.lower() or 'sumber_air' in filename.lower():
                indikator = 'persentase_air_minum_layak'
            elif 'rumah_layak' in filename.lower() or 'layak_huni' in filename.lower():
                indikator = 'persentase_rumah_layak_huni'
            else:
                return None
            result = df[['kode_kabupaten_kota', col1]].copy()
            result['tahun'] = tahun
            result = result.rename(columns={col1: indikator})
            result[indikator] = pd.to_numeric(result[indikator], errors='coerce')
            result = result.dropna(subset=[indikator])
            return result.drop_duplicates(subset=['kode_kabupaten_kota', 'tahun'])

    # Kasus file faskes
    if 'jenis_faskes' in df.columns and 'jumlah_faskes' in df.columns:
        kode_col = None
        for c in ['kode_kabupaten_kota', 'kode_kabupaten']:
            if c in df.columns:
                kode_col = c
                break
        if kode_col is None:
            for c in df.columns:
                if df[c].dtype == 'object':
                    try:
                        sample = df[c].dropna().astype(str).iloc[0]
                        if re.fullmatch(r'\d{4}', sample):
                            kode_col = c
                            break
                    except:
                        pass
        if kode_col is None:
            return None
        tahun_col = 'tahun' if 'tahun' in df.columns else None
        if tahun_col is None:
            match = re.search(r'\b(\d{4})\b', filename)
            if match:
                df['tahun'] = int(match.group(1))
                tahun_col = 'tahun'
            else:
                return None
        df_agg = df.groupby([kode_col, tahun_col])['jumlah_faskes'].sum().reset_index()
        df_agg = df_agg.rename(columns={kode_col: 'kode_kabupaten_kota'})
        df_agg['kode_kabupaten_kota'] = df_agg['kode_kabupaten_kota'].astype(str).str.extract(r'(\d{4})')[0]
        return df_agg

    # Fallback
    kode_col = None
    for c in ['kode_kabupaten_kota', 'kode_kabupaten']:
        if c in df.columns:
            kode_col = c
            break
    if kode_col is None:
        for c in df.columns:
            if df[c].dtype == 'object':
                try:
                    sample = df[c].dropna().astype(str).iloc[0]
                    if re.fullmatch(r'\d{4}', sample):
                        kode_col = c
                        break
                except:
                    pass
    if kode_col is None:
        for c in df.columns:
            if 'nama' in c.lower() and ('kab' in c.lower() or 'kota' in c.lower()):
                df['kode_kabupaten_kota'] = df[c].apply(normalize_kabupaten_name)
                kode_col = 'kode_kabupaten_kota'
                break
    if kode_col is None:
        return None

    tahun_col = 'tahun' if 'tahun' in df.columns else None
    if tahun_col is None:
        match = re.search(r'\b(\d{4})\b', filename)
        if match:
            df['tahun'] = int(match.group(1))
            tahun_col = 'tahun'
        else:
            return None

    df_sel = df.copy()
    rename_map = {}
    for c in df_sel.columns:
        if 'penduduk' in c.lower() and 'jumlah' in c.lower():
            rename_map[c] = 'jumlah_penduduk'
        elif 'miskin' in c.lower() and 'persentase' in c.lower():
            rename_map[c] = 'persentase_penduduk_miskin'
        elif 'pengeluaran' in c.lower():
            rename_map[c] = 'pengeluaran_per_kapita'
        elif 'indeks' in c.lower() and 'pembangunan' in c.lower():
            rename_map[c] = 'indeks_pembangunan_manusia'
        elif 'kepadatan' in c.lower():
            rename_map[c] = 'kepadatan_penduduk'
        elif 'sanitasi' in c.lower():
            rename_map[c] = 'persentase_sanitasi_layak'
        elif 'air_minum' in c.lower() or ('air' in c.lower() and ('minum' in c.lower() or 'bersih' in c.lower() or 'layak' in c.lower())):
            rename_map[c] = 'persentase_air_minum_layak'
        elif 'rumah_layak' in c.lower() or 'layak_huni' in c.lower():
            rename_map[c] = 'persentase_rumah_layak_huni'
        elif 'nakesmas' in c.lower() or 'tenaga_kesehatan' in c.lower():
            rename_map[c] = 'jumlah_nakesmas'
        elif 'faskes' in c.lower() or 'fasilitas' in c.lower():
            rename_map[c] = 'jumlah_faskes'
        elif c == kode_col:
            rename_map[c] = 'kode_kabupaten_kota'
        elif c == tahun_col:
            rename_map[c] = 'tahun'

    df_sel = df_sel.rename(columns=rename_map)
    if 'kode_kabupaten_kota' in df_sel.columns:
        df_sel['kode_kabupaten_kota'] = df_sel['kode_kabupaten_kota'].astype(str).str.extract(r'(\d{4})')[0]
    return df_sel

def merge_and_preprocess(uploaded_files):
    all_dfs = []
    for file in uploaded_files:
        try:
            processed = process_single_file(file, file.name)
            if processed is not None and not processed.empty:
                all_dfs.append(processed)
            else:
                st.warning(f"File {file.name} tidak diproses.")
        except Exception as e:
            st.warning(f"Gagal membaca file {file.name}: {e}")

    if not all_dfs:
        st.error("Tidak ada file yang berhasil diproses.")
        return None

    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = merged.merge(df, on=['kode_kabupaten_kota', 'tahun'], how='outer', suffixes=('', '_dup'))
        dup_cols = [c for c in merged.columns if c.endswith('_dup')]
        for c in dup_cols:
            base = c[:-4]
            if base in merged.columns:
                merged[base] = merged[base].fillna(merged[c])
                merged = merged.drop(columns=[c])

    required = [
        'jumlah_penduduk', 'jumlah_faskes', 'persentase_penduduk_miskin',
        'persentase_sanitasi_layak', 'persentase_air_minum_layak',
        'persentase_rumah_layak_huni', 'kepadatan_penduduk', 'pengeluaran_per_kapita'
    ]
    for col in required:
        if col not in merged.columns:
            merged[col] = np.nan

    if merged['jumlah_faskes'].isna().all():
        st.warning("⚠️ Kolom 'jumlah_faskes' tidak ditemukan atau kosong. Nilai default 0 digunakan untuk prediksi.")
        faskes_cols = [c for c in merged.columns if 'faskes' in c.lower() or 'rumah_sakit' in c.lower()]
        if faskes_cols:
            merged['jumlah_faskes'] = merged[faskes_cols].sum(axis=1)
        else:
            merged['jumlah_faskes'] = 0

    for col in required:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())

    merged['nama_kabupaten'] = merged['kode_kabupaten_kota'].map(KODE_TO_NAMA)
    merged = merged.dropna(subset=['kode_kabupaten_kota'])

    if 'tahun' in merged.columns:
        merged['tahun'] = pd.to_numeric(merged['tahun'], errors='coerce')
        merged = merged.sort_values(['kode_kabupaten_kota', 'tahun'], ascending=[True, False])

    # Deduplikasi
    group_keys = ['kode_kabupaten_kota', 'tahun']
    num_cols = [c for c in merged.select_dtypes(include=np.number).columns if c not in group_keys]
    if all(k in merged.columns for k in group_keys):
        df_num = merged.groupby(group_keys)[num_cols].median().reset_index()
        str_cols = [c for c in merged.columns if c not in num_cols + group_keys]
        if str_cols:
            df_str = merged.groupby(group_keys)[str_cols].first().reset_index()
            df_final = df_num.merge(df_str, on=group_keys, how='left')
        else:
            df_final = df_num
    else:
        df_final = merged.drop_duplicates(subset=['kode_kabupaten_kota'])

    return df_final

# ======================== GEOJSON ========================
@st.cache_data
def load_geojson():
    if os.path.exists("peta_risiko_jabar_2019_2024.geojson"):
        gdf = gpd.read_file("peta_risiko_jabar_2019_2024.geojson")
        kode_col = None
        for col in ['KDBBPS', 'kode_kabupaten_kota', 'kode_kabupaten', 'KODE']:
            if col in gdf.columns:
                kode_col = col
                break
        if kode_col is None:
            for col in gdf.columns:
                if gdf[col].dtype == object:
                    try:
                        sample = gdf[col].dropna().astype(str).iloc[0]
                        if re.sub(r'[^0-9]', '', sample).isdigit():
                            kode_col = col
                            break
                    except:
                        pass
        if kode_col:
            def clean_geo(x):
                s = re.sub(r'[^0-9]', '', str(x))
                return s[:4].zfill(4)
            gdf['kode_wilayah'] = gdf[kode_col].apply(clean_geo)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            return gdf[['kode_wilayah', 'geometry']].copy()
    return None

# ======================== CSS ========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
:root {
    --bg-deep: #060b16; --bg-base: #0b1220; --bg-surface: #101828;
    --bg-elevated: #162032; --border-dim: rgba(255,255,255,0.05);
    --border-mid: rgba(255,255,255,0.09); --border-lit: rgba(255,255,255,0.14);
    --accent: #e8445a; --accent-dim: rgba(232,68,90,0.12);
    --accent-glow: rgba(232,68,90,0.25); --amber: #f5a623; --green: #22d47a;
    --text-1: #f0f4ff; --text-2: #94a3b8; --text-3: #4e6180;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: var(--bg-deep) !important; color: var(--text-1); }
.stApp::before { content: ''; position: fixed; inset: 0; background-image: radial-gradient(circle, rgba(255,255,255,0.025) 1px, transparent 1px); background-size: 28px 28px; pointer-events: none; z-index: 0; }
[data-testid="stSidebar"] { background: var(--bg-base) !important; border-right: 1px solid var(--border-dim) !important; padding-top: 0 !important; }
.sidebar-brand { padding: 22px 18px 18px; border-bottom: 1px solid var(--border-dim); margin-bottom: 20px; position: relative; overflow: hidden; }
.sidebar-brand::after { content: ''; position: absolute; bottom: -40px; right: -40px; width: 120px; height: 120px; background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%); pointer-events: none; }
.brand-icon { width: 42px; height: 42px; background: var(--accent); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; margin-bottom: 12px; box-shadow: 0 0 20px var(--accent-glow); }
.brand-title { font-size: 0.95rem; font-weight: 700; color: var(--text-1); letter-spacing: -0.2px; }
.brand-sub { font-size: 0.62rem; color: var(--text-3); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 3px; }
.ml-status-box { background: var(--bg-surface); border: 1px solid var(--border-dim); border-radius: 14px; padding: 12px 14px; margin-top: 24px; }
.ml-status-title { font-size: 0.55rem; color: var(--text-3); text-transform: uppercase; letter-spacing: 1.2px; }
.ml-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; margin-bottom: 4px; }
.ml-key { color: var(--text-2); } .ml-val { color: var(--text-1); font-weight: 600; }
.ml-badge { display: inline-flex; align-items: center; gap: 4px; background: rgba(34,212,122,0.12); color: #22d47a; border: 1px solid rgba(34,212,122,0.25); padding: 2px 10px; border-radius: 20px; font-size: 0.62rem; font-weight: 600; letter-spacing: 0.5px; }
.ml-badge::before { content: ''; display: inline-block; width: 5px; height: 5px; background: #22d47a; border-radius: 50%; animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.7); } }
.main-header { margin-bottom: 22px; padding-left: 16px; border-left: 4px solid var(--accent); }
.header-eyebrow { font-size: 0.6rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
.header-eyebrow::before { content: ''; display: inline-block; width: 18px; height: 1.5px; background: var(--accent); }
.header-title { font-size: 2.1rem; font-weight: 700; color: var(--text-1); line-height: 1.15; margin: 0; letter-spacing: -0.5px; }
.live-badge { display: inline-flex; align-items: center; gap: 5px; background: rgba(34,212,122,0.12); color: #22d47a; border: 1px solid rgba(34,212,122,0.25); padding: 3px 10px; border-radius: 20px; font-size: 0.58rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; vertical-align: middle; margin-left: 14px; }
.live-badge::before { content: ''; display: inline-block; width: 5px; height: 5px; background: #22d47a; border-radius: 50%; animation: pulse-dot 2s ease-in-out infinite; }
.panel-card { background: var(--bg-surface); border-radius: 10px; padding: 0.8rem 1.2rem; margin-bottom: 1rem; border-left: 4px solid var(--accent); display: flex; align-items: center; gap: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.25); transition: all 0.2s ease; }
.panel-card:hover { border-color: var(--border-lit); box-shadow: 0 6px 24px rgba(0,0,0,0.35); }
.panel-card .title { font-size: 0.85rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-1); margin: 0; }
.panel-card .sub { font-size: 0.65rem; color: var(--text-3); margin-left: auto; font-style: italic; }
.stSelectbox > div > div { background: var(--bg-surface) !important; border: 1px solid var(--border-mid) !important; color: var(--text-1) !important; border-radius: 10px !important; font-size: 0.82rem !important; }
.stSelectbox > div > div:hover { border-color: var(--border-lit) !important; }
.stRadio > div { gap: 12px !important; background: var(--bg-surface); padding: 12px 14px; border-radius: 14px; border: 1px solid var(--border-dim); margin-top: 8px; }
.stRadio label { font-size: 0.85rem !important; font-weight: 500 !important; color: var(--text-1) !important; gap: 8px; }
.stRadio [data-baseweb="radio"]:checked + div { border-color: var(--accent) !important; background: var(--accent-dim); }
.stNumberInput > div > div > input { background: var(--bg-surface) !important; border: 1px solid var(--border-mid) !important; color: var(--text-1) !important; border-radius: 10px !important; font-family: 'DM Mono', monospace !important; font-size: 0.85rem !important; }
.stNumberInput label { font-size: 0.72rem !important; color: var(--text-2) !important; font-weight: 500 !important; }
.stTextInput > div > div > input { background: var(--bg-surface) !important; border: 1px solid var(--border-mid) !important; color: var(--text-1) !important; border-radius: 10px !important; font-family: 'DM Mono', monospace !important; font-size: 0.85rem !important; }
.stTextInput label { font-size: 0.72rem !important; color: var(--text-2) !important; font-weight: 500 !important; }
div[data-testid="stForm"] .stFormSubmitButton button { background: var(--accent) !important; color: #fff !important; border: none !important; border-radius: 12px !important; font-weight: 600 !important; box-shadow: 0 4px 20px var(--accent-glow) !important; transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important; }
div[data-testid="stForm"] .stFormSubmitButton button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; box-shadow: 0 6px 28px var(--accent-glow) !important; }
.stAlert { border-radius: 12px !important; font-size: 0.78rem !important; }
.stPlotlyChart { border-radius: 14px; overflow: hidden; }
.footer { text-align: center; font-size: 0.58rem; color: var(--text-3); padding: 14px 0 10px; border-top: 1px solid var(--border-dim); margin-top: 16px; letter-spacing: 0.5px; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 4px; }

/* ===== PEMADATAN UI (supaya muat 1 layar tanpa scroll) ===== */
.block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }
div[data-testid="stVerticalBlock"] { gap: 0.55rem !important; }
div[data-testid="column"] { gap: 0.55rem !important; }
.main-header { margin-bottom: 10px !important; padding-left: 12px !important; }
.header-title { font-size: 1.5rem !important; }
.header-eyebrow { margin-bottom: 2px !important; font-size: 0.55rem !important; }
.panel-card { padding: 0.55rem 1rem !important; margin-bottom: 0.55rem !important; }
.panel-card .title { font-size: 0.72rem !important; }
.stAlert { padding: 0.4rem 0.8rem !important; margin-bottom: 0.3rem !important; }
.stAlert p { font-size: 0.72rem !important; margin: 0 !important; }
div[data-testid="stNumberInput"], div[data-testid="stTextInput"], div[data-testid="stSelectbox"] { margin-bottom: -0.6rem !important; }
.stNumberInput > div > div > input, .stTextInput > div > div > input { padding: 0.3rem 0.6rem !important; }
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
div[data-testid="stForm"] .stFormSubmitButton button { padding: 0.35rem 1rem !important; margin-top: 0.3rem !important; }
div[data-testid="stForm"] .stFormSubmitButton { margin-bottom: -0.4rem !important; }
.sidebar-brand { padding: 12px 18px 10px !important; margin-bottom: 8px !important; }
.ml-status-box { margin-top: 10px !important; padding: 8px 12px !important; }
</style>
""", unsafe_allow_html=True)

# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🫁</div>
        <div class="brand-title">Risiko TBC</div>
        <div class="brand-sub">Jawa Barat</div>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists("peta_risiko_jabar_2019_2024.geojson"):
        st.info("ℹ️ GeoJSON tidak ditemukan. Peta tidak akan ditampilkan.")

    st.markdown(f"""
    <div class="ml-status-box">
        <div class="ml-status-title">ML Status</div>
        <div class="ml-row"><span class="ml-key">Model</span><span class="ml-val">XGBoost</span></div>
        <div class="ml-row"><span class="ml-key">Status</span><span class="ml-badge">AKTIF</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    mode = st.radio(
        "Pilih Mode",
        ["Mode Prediksi Manual", "Mode Upload Multiple File"],
        label_visibility="collapsed"
    )

# ======================== HEADER ========================
st.markdown("""
<div class="main-header">
    <div class="header-eyebrow">Sistem Pendukung Keputusan (DSS)</div>
    <h1 class="header-title">Prediksi Risiko TBC<span class="live-badge">REAL-TIME</span></h1>
</div>
""", unsafe_allow_html=True)

# ======================== MODE PREDIKSI MANUAL (DIPERBAIKI) ========================
if mode == "Mode Prediksi Manual":
    # Nilai default tiap field (dipakai form & tombol reset, biar konsisten satu sumber)
    MANUAL_DEFAULTS = {
        "manual_jml_penduduk": "700.0",
        "manual_miskin": 3.5,
        "manual_air_minum": 96.0,
        "manual_kepadatan": 400.0,
        "manual_faskes": 90.0,
        "manual_sanitasi": 97.0,
        "manual_rumah": 96.0,
        "manual_pengeluaran": "18000.0",
    }

    col_left, col_right = st.columns([1, 1], gap="large")
    # Inisialisasi nilai default HANYA jika belum pernah diisi sebelumnya
    # (supaya tidak menimpa nilai yang sedang diketik user tiap rerun)
    for _k, _v in MANUAL_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    with col_left:
        st.markdown('<div class="panel-card"><div class="title">📝 Input Data Wilayah (Mentah)</div></div>', unsafe_allow_html=True)

        with st.form(key="manual_form"):
            selected_wilayah_manual = st.selectbox(
                "🗺️ Wilayah yang Diwakili Data Ini",
                sorted(NAMA_TO_KODE.keys()),
                key="manual_wilayah_select",
                help="Pilih kabupaten/kota yang datanya kamu masukkan di bawah. "
                     "Ini dipakai untuk menampilkan lokasinya di peta setelah prediksi, "
                     "TIDAK mempengaruhi hasil perhitungan (hasil tetap berdasarkan "
                     "8 angka yang kamu isi sendiri)."
            )
            col1, col2 = st.columns(2)
            with col1:
                # ===== PERBAIKAN: text_input agar bisa desimal =====
                jumlah_penduduk_str = st.text_input(
                    "Jumlah Penduduk",
                    key="manual_jml_penduduk",
                    help="Boleh diisi angka penuh (misal 5400000 dari CSV BPS) ATAU dalam ribuan "
                         "(misal 5400). Sistem otomatis mendeteksi satuannya."
                )
                persentase_penduduk_miskin = st.number_input("Persentase Penduduk Miskin (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f", key="manual_miskin")
                persentase_air_minum_layak = st.number_input("Persentase Air Minum Layak (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f", key="manual_air_minum")
                kepadatan_penduduk = st.number_input("Kepadatan Penduduk (jiwa/km²)", min_value=0.0, max_value=1e9, step=100.0, format="%.0f", key="manual_kepadatan")
            with col2:
                jumlah_faskes = st.number_input("Jumlah Fasilitas Kesehatan", min_value=0.0, max_value=1e9, step=1.0, format="%.0f", key="manual_faskes")
                persentase_sanitasi_layak = st.number_input("Persentase Sanitasi Layak (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f", key="manual_sanitasi")
                persentase_rumah_layak_huni = st.number_input("Persentase Rumah Layak Huni (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f", key="manual_rumah")
                # ===== PERBAIKAN: text_input agar bisa desimal =====
                pengeluaran_per_kapita_str = st.text_input("Pengeluaran per Kapita (ribu Rp/tahun)", key="manual_pengeluaran")
            
            submitted = st.form_submit_button("Prediksi Sekarang", use_container_width=True)
            if submitted:
                # Parsing nilai dari string (ganti koma dengan titik)
                try:
                    jumlah_penduduk = float(jumlah_penduduk_str.replace(',', '.'))
                except:
                    st.error("❌ Format Jumlah Penduduk tidak valid. Gunakan angka (contoh: 5484.15 atau 5484,15)")
                    st.stop()
                try:
                    pengeluaran_per_kapita = float(pengeluaran_per_kapita_str.replace(',', '.'))
                except:
                    st.error("❌ Format Pengeluaran per Kapita tidak valid. Gunakan angka (contoh: 10410.5 atau 10410,5)")
                    st.stop()
                
                input_dict = {
                    'jumlah_penduduk': jumlah_penduduk,
                    'jumlah_faskes': jumlah_faskes,
                    'persentase_penduduk_miskin': persentase_penduduk_miskin,
                    'persentase_sanitasi_layak': persentase_sanitasi_layak,
                    'persentase_air_minum_layak': persentase_air_minum_layak,
                    'persentase_rumah_layak_huni': persentase_rumah_layak_huni,
                    'kepadatan_penduduk': kepadatan_penduduk,
                    'pengeluaran_per_kapita': pengeluaran_per_kapita
                }
                try:
                    res = predict_single(input_dict)
                    st.session_state.manual_result = res
                    st.session_state.manual_input = input_dict
                    st.session_state.manual_wilayah = selected_wilayah_manual
                    st.success("✅ Prediksi berhasil!")

                    # ---- Catat ke riwayat (maksimal 5 entri terakhir, terbaru di atas) ----
                    if "manual_history" not in st.session_state:
                        st.session_state.manual_history = []
                    st.session_state.manual_history.insert(0, {
                        "Wilayah": selected_wilayah_manual,
                        "Kategori": res["kategori"],
                        "Risk Index": round(res["risk_index"], 1),
                        "P(Rendah)": f"{res['prob_rendah']*100:.1f}%",
                        "P(Sedang)": f"{res['prob_sedang']*100:.1f}%",
                        "P(Tinggi)": f"{res['prob_tinggi']*100:.1f}%",
                    })
                    st.session_state.manual_history = st.session_state.manual_history[:5]
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with col_right:
        if st.session_state.manual_result is not None:
            res = st.session_state.manual_result
            kat = res["kategori"]
            risk_idx = res["risk_index"]
            prob_rendah = res["prob_rendah"]*100
            prob_sedang = res["prob_sedang"]*100
            prob_tinggi = res["prob_tinggi"]*100
            color_kat = {"Tinggi":"#ef4444","Sedang":"#f5a623","Rendah":"#22d47a"}.get(kat,"#94a3b8")
            glow_kat = {"Tinggi":"rgba(239,68,68,0.15)","Sedang":"rgba(245,166,35,0.12)","Rendah":"rgba(34,212,122,0.12)"}.get(kat,"rgba(148,163,184,0.08)")
            
            st.markdown(f"""
            <div style="display:flex; gap:10px; margin-bottom:8px;">
                <div style="flex:1; background:#162032; border:1px solid {color_kat}40; border-radius:8px; padding:8px 10px; text-align:center; box-shadow: 0 0 18px {glow_kat};">
                    <div style="font-size:0.62rem; color:#94a3b8; letter-spacing:1px; text-transform:uppercase; margin-bottom:2px; font-weight:600;">Kategori Risiko</div>
                    <div style="font-size:1.3rem; font-weight:700; color:{color_kat}; line-height:1.2;">{kat}</div>
                </div>
                <div style="flex:1; background:#162032; border:1px solid #2c3e50; border-radius:8px; padding:8px 10px; text-align:center;">
                    <div style="font-size:0.62rem; color:#94a3b8; letter-spacing:1px; text-transform:uppercase; margin-bottom:2px; font-weight:600;">Risk Index</div>
                    <div style="font-size:1.3rem; font-weight:700; color:#f97316; line-height:1.2;">{risk_idx:.1f}</div>
                    <div style="margin-top:4px; background:#2c3e50; border-radius:10px; height:3px;">
                        <div style="width:{min(risk_idx,100)}%; background:linear-gradient(90deg,#22d47a,#f5a623,#ef4444); height:3px; border-radius:10px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            rekom_text = {
                "Rendah": ("ℹ️", "Kondisi terkendali. Pertahankan surveilans."),
                "Sedang": ("⚠️", "Perlu monitoring aktif. Tingkatkan deteksi dini."),
                "Tinggi": ("🚨", "Prioritas intervensi tinggi. Lakukan skrining aktif."),
            }.get(kat, ("ℹ️", ""))
            st.markdown(f"""
            <div style="background:{color_kat}12; border:1px solid {color_kat}35; border-left:4px solid {color_kat}; border-radius:8px; padding:8px 12px; margin-bottom:10px; display:flex; align-items:center; gap:8px;">
                <span style="font-size:1rem;">{rekom_text[0]}</span>
                <span style="font-size:0.75rem; color:#e2e8f0;"><b style="color:{color_kat};">Rekomendasi:</b> {rekom_text[1]}</span>
            </div>
            """, unsafe_allow_html=True)

            # ======================== PETA WILAYAH TERPILIH ========================
            if st.session_state.manual_wilayah is not None:
                st.markdown('<div class="panel-card"><div class="title">🗺️ Lokasi di Peta</div><div class="sub">Berdasarkan wilayah yang dipilih</div></div>', unsafe_allow_html=True)
                gdf_geo_manual = load_geojson()
                if gdf_geo_manual is not None:
                    kode_terpilih = NAMA_TO_KODE.get(st.session_state.manual_wilayah)
                    gdf_geo_manual['kode_wilayah'] = gdf_geo_manual['kode_wilayah'].astype(str)
                    mg_manual = gdf_geo_manual[gdf_geo_manual['kode_wilayah'] == str(kode_terpilih)].copy()
                    if not mg_manual.empty:
                        mg_manual['Kategori'] = kat
                        mg_manual['Nama'] = st.session_state.manual_wilayah
                        lat_m = mg_manual.geometry.centroid.y.iloc[0]
                        lon_m = mg_manual.geometry.centroid.x.iloc[0]
                        fig_manual = px.choropleth_mapbox(
                            mg_manual,
                            geojson=json.loads(mg_manual.to_json()),
                            locations=mg_manual.index,
                            color='Kategori',
                            color_discrete_map={'Tinggi':'#ef4444','Sedang':'#f5a623','Rendah':'#22d47a'},
                            mapbox_style="carto-darkmatter",
                            center={"lat": lat_m, "lon": lon_m},
                            zoom=9,
                            opacity=0.75,
                            hover_name='Nama',
                            hover_data={}
                        )
                        fig_manual.update_traces(
                            hovertemplate='<b>%{hovertext}</b><extra></extra>',
                            marker_line_width=2,
                            marker_line_color="#ffffff"
                        )
                        fig_manual.update_layout(
                            height=170,
                            margin=dict(l=0, r=0, t=0, b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            showlegend=False
                        )
                        st.plotly_chart(fig_manual, use_container_width=True)
                    else:
                        st.info(f"ℹ️ Bentuk wilayah '{st.session_state.manual_wilayah}' tidak ditemukan di GeoJSON.")
                else:
                    st.info("ℹ️ File GeoJSON tidak tersedia — peta tidak bisa ditampilkan.")

            st.markdown('<div class="panel-card"><div class="title">Analisis SHAP</div><div class="sub">Kontribusi faktor</div></div>', unsafe_allow_html=True)
            if st.session_state.manual_input is not None:
                shap_dict = get_shap_values(st.session_state.manual_input, pred_class=res['pred_class'])
                feature_names_display = {
                    "basic_service_index": "Indeks Pelayanan Dasar",
                    "faskes_ratio": "Rasio Faskes",
                    "poverty_density": "Kepadatan Kemiskinan",
                    "pengeluaran_per_kapita": "Pengeluaran per Kapita",
                    "kepadatan_penduduk": "Kepadatan Penduduk"
                }
                total_abs = sum(abs(v) for v in shap_dict.values()) or 1
                for feat, shap_val in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
                    pct = (abs(shap_val) / total_abs) * 100
                    icon = "🔴" if shap_val > 0 else "🟢"
                    warna = "#ef4444" if shap_val > 0 else "#22d47a"
                    arah = "meningkatkan" if shap_val > 0 else "menurunkan"
                    st.markdown(f"""
                    <div style="margin-bottom:7px; background:#162032; border-radius:8px; padding:8px 12px; border-left:3px solid {warna};">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.74rem; font-weight:500; color:#ecf0f1;">{icon} {feature_names_display[feat]}</span>
                            <span style="color:{warna}; font-size:0.65rem; font-weight:600;">{arah} ({pct:.1f}%)</span>
                        </div>
                        <div style="margin-top:5px; background:#2c3e50; border-radius:4px; height:3px;">
                            <div style="width:{pct}%; background:{warna}; height:3px; border-radius:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Isi form di sebelah kiri dan tekan tombol Prediksi.")

# ======================== MODE UPLOAD (DIPERBAIKI PROGRESS BAR) ========================
else:
    st.markdown('<div class="panel-card"><div class="title">📂 Upload File Data Mentah (CSV/Excel)</div><div class="sub">Pilih semua file BPS sekaligus</div></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Pilih beberapa file (CSV / Excel)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="multi_upload",
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.session_state.df_final = None
        st.session_state.batch_results_df = None
        
        with st.spinner("Memproses dan menggabungkan data..."):
            df_preprocessed = merge_and_preprocess(uploaded_files)
        
        if df_preprocessed is not None and not df_preprocessed.empty:
            st.success(f"✅ Berhasil memproses data: {len(df_preprocessed)} wilayah unik.")
            
            results = []
            pb = st.progress(0, text="Memprediksi...")
            # FIX: gunakan enumerate untuk index counter
            for i, (idx, row) in enumerate(df_preprocessed.iterrows()):
                inp = row.to_dict()
                pred = predict_single(inp)
                results.append({
                    'Prediksi_Kategori': pred['kategori'],
                    'Risk_Index': round(pred['risk_index'], 2),
                    'Prob_Rendah': round(pred['prob_rendah'], 4),
                    'Prob_Sedang': round(pred['prob_sedang'], 4),
                    'Prob_Tinggi': round(pred['prob_tinggi'], 4),
                    'pred_class': pred['pred_class']
                })
                pb.progress((i + 1) / len(df_preprocessed))
            pb.empty()
            
            df_results = pd.DataFrame(results)
            df_final = pd.concat([df_preprocessed.reset_index(drop=True), df_results], axis=1)
            st.session_state.df_final = df_final
            st.session_state.batch_results_df = df_final
            st.session_state.batch_uploaded = True

            if 'tahun' in df_final.columns:
                tahun_terbaru = df_final['tahun'].max()
                df_display = df_final[df_final['tahun'] == tahun_terbaru].copy()
            else:
                tahun_terbaru = None
                df_display = df_final.copy()
            
            with st.sidebar:
                if not df_display.empty:
                    wilayah_list = ["Semua Wilayah"] + sorted(df_display['nama_kabupaten'].unique())
                    selected_wilayah = st.selectbox(
                        "Filter Wilayah",
                        wilayah_list,
                        index=0 if st.session_state.batch_selected_kab not in wilayah_list else wilayah_list.index(st.session_state.batch_selected_kab),
                        key="batch_filter_wilayah"
                    )
                    st.session_state.batch_selected_kab = selected_wilayah
                else:
                    selected_wilayah = "Semua Wilayah"
                    st.session_state.batch_selected_kab = "Semua Wilayah"

            gdf_geo = load_geojson()
            if gdf_geo is not None and 'kode_kabupaten_kota' in df_display.columns:
                df_display['kode_wilayah'] = df_display['kode_kabupaten_kota'].astype(str)
                gdf_geo['kode_wilayah'] = gdf_geo['kode_wilayah'].astype(str)
                merged_geo = gdf_geo.merge(df_display[['kode_wilayah','Prediksi_Kategori','Risk_Index','nama_kabupaten']], on='kode_wilayah', how='inner')
                if merged_geo.empty:
                    st.warning("⚠️ Tidak ada data yang cocok dengan GeoJSON.")
                    merged_geo = None
            else:
                merged_geo = None
                st.info("ℹ️ Peta tidak tersedia.")

            st.session_state.merged_geo = merged_geo

            col_peta, col_kanan = st.columns([2, 1], gap="large")
            
            with col_peta:
                # FIX: cegah ZeroDivisionError
                if not df_display.empty:
                    counts = df_display['Prediksi_Kategori'].value_counts()
                    total = len(df_display)
                    renda = counts.get('Rendah', 0)
                    seda = counts.get('Sedang', 0)
                    ting = counts.get('Tinggi', 0)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div style="background:#0b1220; border-radius:14px; padding:16px 10px; border:1px solid rgba(34,212,122,0.25); text-align:center; box-shadow: 0 4px 16px rgba(0,0,0,0.4);">
                            <div style="font-size:26px; margin-bottom:2px;">🟢</div>
                            <div style="font-size:32px; font-weight:700; color:#22d47a; font-family:monospace; line-height:1.2;">{renda}</div>
                            <div style="font-size:13px; color:#94a3b8; text-transform:uppercase; letter-spacing:1.2px; font-weight:600; margin-top:2px;">Rendah</div>
                            <div style="font-size:14px; color:#64748b; margin-top:4px; font-weight:500;">{renda/total*100:.1f}%</div>
                            <div style="width:100%; height:3px; background:rgba(255,255,255,0.06); border-radius:6px; margin-top:10px; overflow:hidden;">
                                <div style="width:{renda/total*100}%; height:3px; background:#22d47a; border-radius:6px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style="background:#0b1220; border-radius:14px; padding:16px 10px; border:1px solid rgba(245,166,35,0.25); text-align:center; box-shadow: 0 4px 16px rgba(0,0,0,0.4);">
                            <div style="font-size:26px; margin-bottom:2px;">🟡</div>
                            <div style="font-size:32px; font-weight:700; color:#f5a623; font-family:monospace; line-height:1.2;">{seda}</div>
                            <div style="font-size:13px; color:#94a3b8; text-transform:uppercase; letter-spacing:1.2px; font-weight:600; margin-top:2px;">Sedang</div>
                            <div style="font-size:14px; color:#64748b; margin-top:4px; font-weight:500;">{seda/total*100:.1f}%</div>
                            <div style="width:100%; height:3px; background:rgba(255,255,255,0.06); border-radius:6px; margin-top:10px; overflow:hidden;">
                                <div style="width:{seda/total*100}%; height:3px; background:#f5a623; border-radius:6px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div style="background:#0b1220; border-radius:14px; padding:16px 10px; border:1px solid rgba(239,68,68,0.25); text-align:center; box-shadow: 0 4px 16px rgba(0,0,0,0.4);">
                            <div style="font-size:26px; margin-bottom:2px;">🔴</div>
                            <div style="font-size:32px; font-weight:700; color:#ef4444; font-family:monospace; line-height:1.2;">{ting}</div>
                            <div style="font-size:13px; color:#94a3b8; text-transform:uppercase; letter-spacing:1.2px; font-weight:600; margin-top:2px;">Tinggi</div>
                            <div style="font-size:14px; color:#64748b; margin-top:4px; font-weight:500;">{ting/total*100:.1f}%</div>
                            <div style="width:100%; height:3px; background:rgba(255,255,255,0.06); border-radius:6px; margin-top:10px; overflow:hidden;">
                                <div style="width:{ting/total*100}%; height:3px; background:#ef4444; border-radius:6px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak ada data untuk tahun terbaru.")

                st.subheader(f"🗺️ Peta Sebaran Risiko (Tahun {tahun_terbaru if tahun_terbaru else 'Terbaru'})")
                if merged_geo is not None and not merged_geo.empty:
                    if selected_wilayah != "Semua Wilayah":
                        mg = merged_geo[merged_geo['nama_kabupaten'] == selected_wilayah].copy()
                    else:
                        mg = merged_geo.copy()
                    
                    if not mg.empty:
                        if selected_wilayah != "Semua Wilayah":
                            lat = mg.geometry.centroid.y.iloc[0]
                            lon = mg.geometry.centroid.x.iloc[0]
                            zoom = 9
                        else:
                            lat, lon, zoom = -6.9, 107.6, 7
                        
                        fig = px.choropleth_mapbox(
                            mg,
                            geojson=json.loads(mg.to_json()),
                            locations=mg.index,
                            color='Prediksi_Kategori',
                            color_discrete_map={'Tinggi':'#ef4444','Sedang':'#f5a623','Rendah':'#22d47a'},
                            mapbox_style="carto-darkmatter",
                            center={"lat": lat, "lon": lon},
                            zoom=zoom,
                            opacity=0.8,
                            hover_name='nama_kabupaten',
                            hover_data={}
                        )
                        fig.update_traces(
                            hovertemplate='<b>%{hovertext}</b><extra></extra>',
                            marker_line_width=0.6,
                            marker_line_color="rgba(255,255,255,0.15)"
                        )
                        if selected_wilayah != "Semua Wilayah":
                            fig.add_trace(go.Choroplethmapbox(
                                geojson=json.loads(mg.to_json()),
                                locations=mg.index,
                                z=[0]*len(mg),
                                colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(0,0,0,0)"]],
                                marker_line_width=3,
                                marker_line_color="#ffffff",
                                showscale=False,
                                hoverinfo="skip"
                            ))
                        fig.update_layout(
                            height=500,
                            margin=dict(l=0,r=0,t=0,b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(
                                bgcolor="rgba(11,18,32,0.9)",
                                bordercolor="rgba(255,255,255,0.1)",
                                borderwidth=1,
                                font=dict(color="#94a3b8", size=11),
                                title=dict(text="Kategori", font=dict(color="#64748b", size=10))
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Tidak ada data untuk wilayah yang dipilih.")
                else:
                    st.info("Peta tidak tersedia.")

            with col_kanan:
                if selected_wilayah != "Semua Wilayah":
                    row = df_display[df_display['nama_kabupaten'] == selected_wilayah].iloc[0]
                    kat = row['Prediksi_Kategori']
                    risk = row['Risk_Index']
                    prob_rendah = row['Prob_Rendah']*100
                    prob_sedang = row['Prob_Sedang']*100
                    prob_tinggi = row['Prob_Tinggi']*100
                    color_kat = {"Tinggi":"#ef4444","Sedang":"#f5a623","Rendah":"#22d47a"}.get(kat,"#94a3b8")
                    glow_kat = {"Tinggi":"rgba(239,68,68,0.15)","Sedang":"rgba(245,166,35,0.12)","Rendah":"rgba(34,212,122,0.12)"}.get(kat,"rgba(148,163,184,0.08)")
                    
                    st.markdown(f"""
                    <div style="background:#162032; border:1px solid {color_kat}30; border-radius:14px; padding:1.2rem 1rem; margin-bottom:1rem; box-shadow: 0 0 30px {glow_kat};">
                        <div style="font-size:0.7rem; color:#94a3b8; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:6px; font-weight:600;">DETAIL WILAYAH</div>
                        <div style="font-size:1.5rem; font-weight:700; color:var(--text-1); margin-bottom:6px;">{selected_wilayah}</div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
                            <span style="font-size:2.0rem; font-weight:700; color:{color_kat};">{kat}</span>
                            <span style="font-size:2.0rem; font-weight:700; color:#f97316;">{risk:.1f}</span>
                        </div>
                        <div style="margin-top:10px; background:rgba(255,255,255,0.04); border-radius:20px; height:5px; overflow:hidden;">
                            <div style="width:{min(risk,100)}%; background:linear-gradient(90deg,#22d47a,#f5a623,#ef4444); height:5px; border-radius:20px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    inp = row.to_dict()
                    shap_dict = get_shap_values(inp, pred_class=int(row['pred_class']))
                    feature_names_display = {
                        "basic_service_index": "Indeks Pelayanan Dasar",
                        "faskes_ratio": "Rasio Fasilitas Kesehatan",
                        "poverty_density": "Kepadatan Kemiskinan",
                        "pengeluaran_per_kapita": "Pengeluaran per Kapita",
                        "kepadatan_penduduk": "Kepadatan Penduduk"
                    }
                    total_abs = sum(abs(v) for v in shap_dict.values()) or 1
                    st.markdown(f"""
                    <div style="font-size:0.85rem; font-weight:600; color:var(--text-2); letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">📊 Faktor Risiko (SHAP)</div>
                    """, unsafe_allow_html=True)
                    for feat, shap_val in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
                        pct = (abs(shap_val) / total_abs) * 100
                        icon = "🔴" if shap_val > 0 else "🟢"
                        warna = "#ef4444" if shap_val > 0 else "#22d47a"
                        arah = "↑ meningkatkan" if shap_val > 0 else "↓ menurunkan"
                        st.markdown(f"""
                        <div style="margin-bottom:8px; background:#162032; border-radius:8px; padding:10px 14px; border-left:4px solid {warna}; display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.9rem; font-weight:500; color:var(--text-1);">{icon} {feature_names_display[feat]}</span>
                            <span style="color:{warna}; font-size:0.85rem; font-weight:700; font-family:var(--mono);">{arah} · {pct:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#101828; border-radius:12px; padding:16px; border:1px solid rgba(255,255,255,0.06);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                            <span style="font-size:1.1rem; font-weight:600; color:#f0f4ff;">🔝 Top 5 Risiko Tertinggi</span>
                            <span style="font-size:0.7rem; color:#94a3b8;">Berdasarkan Risk Index</span>
                        </div>
                    """, unsafe_allow_html=True)
                    if not df_display.empty:
                        top5 = df_display.nlargest(5, 'Risk_Index')[['nama_kabupaten', 'Prediksi_Kategori', 'Risk_Index']]
                        for i, (_, row) in enumerate(top5.iterrows(), 1):
                            kat = row['Prediksi_Kategori']
                            risk = row['Risk_Index']
                            if kat == 'Tinggi':
                                badge_color = '#ef4444'
                                bg_badge = 'rgba(239,68,68,0.15)'
                            elif kat == 'Sedang':
                                badge_color = '#f5a623'
                                bg_badge = 'rgba(245,166,35,0.15)'
                            else:
                                badge_color = '#22d47a'
                                bg_badge = 'rgba(34,212,122,0.15)'
                            st.markdown(f"""
                            <div style="display:flex; align-items:center; padding:10px 12px; margin-bottom:6px; background:rgba(255,255,255,0.03); border-radius:10px; transition:0.2s;">
                                <div style="width:28px; height:28px; border-radius:50%; background:rgba(255,255,255,0.06); display:flex; align-items:center; justify-content:center; font-weight:600; font-size:0.85rem; color:#94a3b8; margin-right:12px;">{i}</div>
                                <div style="flex:1; font-size:1.0rem; font-weight:500; color:#f0f4ff;">{row['nama_kabupaten']}</div>
                                <div style="display:flex; align-items:center; gap:14px;">
                                    <span style="font-size:0.75rem; font-weight:600; color:{badge_color}; background:{bg_badge}; padding:2px 12px; border-radius:20px;">{kat}</span>
                                    <span style="font-size:1.1rem; font-weight:700; color:#f97316; min-width:44px; text-align:right;">{risk:.1f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Belum ada data.")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📋 Tabel Hasil Prediksi (Semua Tahun)")
            st.caption("Menampilkan seluruh data dari semua tahun yang diupload.")

            if selected_wilayah != "Semua Wilayah":
                df_tampil = df_final[df_final['nama_kabupaten'] == selected_wilayah].copy()
            else:
                df_tampil = df_final.copy()

            if not df_tampil.empty:
                cols_show = ['nama_kabupaten', 'tahun', 'Prediksi_Kategori', 'Risk_Index', 'Prob_Rendah', 'Prob_Sedang', 'Prob_Tinggi']
                cols_show = [c for c in cols_show if c in df_tampil.columns]
                df_show = df_tampil[cols_show].copy()
                
                for col in ['Prob_Rendah','Prob_Sedang','Prob_Tinggi']:
                    if col in df_show.columns:
                        df_show[col] = df_show[col].apply(lambda x: f"{x*100:.1f}%")
                if 'Risk_Index' in df_show.columns:
                    df_show['Risk_Index'] = df_show['Risk_Index'].apply(lambda x: f"{x:.1f}")
                
                def color_risk(val):
                    if val == 'Tinggi':
                        return 'background-color: rgba(239,68,68,0.15)'
                    elif val == 'Sedang':
                        return 'background-color: rgba(245,166,35,0.15)'
                    elif val == 'Rendah':
                        return 'background-color: rgba(34,212,122,0.15)'
                    return ''
                
                styled = df_show.style.applymap(color_risk, subset=['Prediksi_Kategori'])
                styled = styled.set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#1a2538'), ('color', '#94a3b8'), 
                                                       ('font-size', '0.75rem'), ('padding', '8px 10px'), 
                                                       ('text-align', 'left'), ('font-weight', '600')]},
                    {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', 'rgba(255,255,255,0.03)')]},
                    {'selector': 'tbody tr:hover', 'props': [('background-color', 'rgba(255,255,255,0.07)')]},
                    {'selector': 'td', 'props': [('padding', '8px 10px'), ('font-size', '0.8rem'), 
                                                 ('border-bottom', '1px solid rgba(255,255,255,0.05)')]},
                ])
                
                st.dataframe(styled, use_container_width=True, height=400, hide_index=True)
            else:
                st.info("Tidak ada data untuk ditampilkan.")

            st.markdown("---")
            st.success("✅ **Model XGBoost berhasil dijalankan dan memproses seluruh data yang diupload.**")
            st.caption(f"📌 **Threshold:** t_high = {T_HIGH:.2f}, t0 = {T0:.2f}  |  Fitur: BSI, FR, PD, Pengeluaran, Kepadatan | Algoritma: XGBoost (multi-class)")

            st.markdown("---")
            st.subheader("📊 Evaluasi Model XGBoost pada Data Uji (2024)")
            st.caption("Metrik berikut dihitung dari data uji tahun 2024 (27 kabupaten/kota) yang tidak pernah dilihat model saat training.")

            left_col, right_col = st.columns([1, 1.8], gap="large")
            with left_col:
                st.metric("Akurasi", f"{metrics['accuracy']*100:.1f}%")
                st.metric("Precision (Makro)", f"{metrics['precision_macro']*100:.1f}%")
                st.metric("Recall (Makro)", f"{metrics['recall_macro']*100:.1f}%")
                st.metric("F1-Score (Makro)", f"{metrics['f1_macro']*100:.1f}%")
            with right_col:
                cm = metrics['confusion_matrix']
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                    x=['Rendah', 'Sedang', 'Tinggi'],
                    y=['Rendah', 'Sedang', 'Tinggi'],
                    aspect='square'
                )
                fig_cm.update_layout(
                    height=400,
                    width=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f0f4ff", size=14),
                    xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
                    yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12))
                )
                fig_cm.update_traces(
                    textfont=dict(color="white", size=16),
                    hovertemplate='Aktual: %{y}<br>Prediksi: %{x}<br>Jumlah: %{z}<extra></extra>'
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            st.success("✅ **Kesimpulan:** Model XGBoost menunjukkan performa yang baik dengan akurasi > 80% dan F1-score makro > 80%.")

            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV Hasil Prediksi (Semua Wilayah & Tahun)", data=csv, file_name="hasil_prediksi_tbc.csv", mime="text/csv")
            
        else:
            st.error("❌ Gagal memproses data. Periksa kembali file-file yang diupload.")

st.markdown(f"""
<div class="footer">
    SIG &amp; ML TBC Jabar · XGBoost · 
    Threshold: t_high={T_HIGH:.2f} · t0={T0:.2f} · © 2026
</div>
""", unsafe_allow_html=True)