import streamlit as st
import folium
import numpy as np
import base64
from io import BytesIO
from PIL import Image # Pillow 라이브러리 추가
import folium.plugins # Fullscreen 플러그인 사용을 위해 추가

# --- 추가해야 할 임포트 ---
from streamlit_folium import folium_static 
# -------------------------

# --- 1. Streamlit 앱 설정 ---
st.set_page_config(layout="wide")
st.title("🌊 미래 해수면 상승 시뮬레이션 (글로벌 예측)")
st.markdown("---")

st.sidebar.header("설정")
selected_year = st.sidebar.slider(
    "미래 연도 선택",
    min_value=2024,
    max_value=2100,
    value=2050,
    step=10,
    help="선택한 연도에 따른 예상 해수면 상승량"
)

# IPCC 보고서 기반의 대략적인 해수면 상승 예측 (센티미터)
# 이 수치는 RCP2.6 (저배출) ~ RCP8.5 (고배출) 시나리오 범위의 중간값을 임의로 설정한 것입니다.
# 실제 프로젝트에서는 더 정확한 데이터를 찾아 적용해야 합니다.
sea_level_rise_cm = {
    2024: 0,
    2030: 5,  # 예시
    2040: 10, # 예시
    2050: 20, # 2100년까지 최대 1미터 상승 시나리오의 중간값 정도로 가정
    2060: 30, # 예시
    2070: 45, # 예시
    2080: 60, # 예시
    2090: 75, # 예시
    2100: 100 # 최대 1미터 (100cm) 상승으로 가정 (고배출 시나리오 근사치)
}

# 선택된 연도에 따른 해수면 상승량 계산 (선형 보간)
rise_amount_cm = np.interp(
    selected_year,
    list(sea_level_rise_cm.keys()),
    list(sea_level_rise_cm.values())
)
rise_amount_m = rise_amount_cm / 100.0 # 미터 단위로 변환

st.sidebar.metric(label="예상 해수면 상승량", value=f"{rise_amount_cm:.1f} cm ({rise_amount_m:.2f} m)")
st.sidebar.markdown(
    """
    <small>이 수치는 IPCC 보고서 기반의 대략적인 예측이며, 실제와 다를 수 있습니다.</small>
    <small>정확한 데이터는 [IPCC](https://www.ipcc.ch/report/ar6/wg1/) 및 [NASA](https://sealevel.nasa.gov/data/data-portal) 등에서 확인해주세요.</small>
    """,
    unsafe_allow_html=True
)

# --- 2. 가상의 고도 데이터 생성 (실제 데이터 대신) ---
# 실제 데이터 사용 시, 이 부분을 DEM 파일 로드 로직으로 대체해야 합니다.
# 예: rasterio.open('dem_file.tif').read(1)
@st.cache_data
def generate_fake_dem_data(resolution=150): # 해상도를 기본값으로 설정하여 재사용
    # 전 세계 대략적인 위경도 범위
    lat_min, lat_max = -90, 90
    lon_min, lon_max = -180, 180

    # 간단한 고도 데이터 생성 (대륙처럼 보이는 패턴)
    lats = np.linspace(lat_min, lat_max, resolution)
    lons = np.linspace(lon_min, lon_max, resolution * 2) # 경도 범위 2배
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 대략적인 고도 패턴 (높은 대륙, 낮은 해안선)
    # 중앙 부분은 높게, 가장자리로 갈수록 낮게
    # 실제 DEM 파일은 0 이하가 바다, 0 이상이 육지
    dem_data = 100 * np.exp(-((lon_grid/50)**2 + (lat_grid/30)**2)) # 중앙 고지대
    dem_data += 50 * np.exp(-(((lon_grid-100)/70)**2 + ((lat_grid+50)/40)**2)) # 또 다른 고지대
    dem_data -= 20 # 전체적으로 낮춰서 일부가 바다처럼 보이게

    # 음수 값을 0으로 만들고 (바다), 양수 값은 육지 고도
    dem_data[dem_data < 0] = 0
    return dem_data, (lat_min, lon_min, lat_max, lon_max) # bounds도 함께 반환

dem_data, bounds = generate_fake_dem_data() # 해상도 조절

# --- 3. 침수 지역 계산 및 지도 생성 ---
def create_flood_map(dem_array, current_sea_level_m, map_bounds):
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="cartodbpositron", control_scale=True)

    # 침수될 영역 (현재 해수면 이하의 고도)
    flooded_area_mask = dem_array < current_sea_level_m

    # 침수 영역을 이미지 오버레이로 표현
    if np.any(flooded_area_mask):
        # 마스크를 RGBA 이미지로 변환 (푸른색, 반투명)
        # 0: 투명, 1: 푸른색 (침수)
        img_data = np.zeros((*flooded_area_mask.shape, 4), dtype=np.uint8)
        img_data[flooded_area_mask] = [0, 0, 255, 128]  # 파란색, 50% 투명도

        # 이미지를 BytesIO에 저장 후 base64 인코딩
        img_buffer = BytesIO()
        img = Image.fromarray(img_data, 'RGBA')
        img.save(img_buffer, format="PNG")
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

        # Folium에 이미지 오버레이 추가
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[map_bounds[0], map_bounds[1]], [map_bounds[2], map_bounds[3]]],
            opacity=1.0, # 마스크 이미지 자체는 투명도를 가졌으므로, 오버레이는 불투명
            name=f"Flooded Area ({current_sea_level_m:.2f}m rise)"
        ).add_to(m)
        
    folium.LayerControl().add_to(m)
    return m

# 지도 생성
flood_map = create_flood_map(dem_data, rise_amount_m, bounds)

# --- 4. Streamlit에 지도 표시 ---
st.subheader("예측된 해수면 상승으로 인한 침수 지역")
st.markdown(
    """
    **파란색 반투명 영역**은 선택된 연도의 예상 해수면 상승량에 따라 침수될 것으로 예측되는 지역입니다.
    가상의 고도 데이터를 기반으로 하므로 실제 지형과는 다를 수 있습니다.
    """
)
# Fullscreen 플러그인을 지도에 추가
folium.plugins.Fullscreen(
    position="topright",
    title="전체 화면",
    title_action="전체 화면 종료",
    # force_separate_button=True # 별도 버튼 강제 여부 (옵션)
).add_to(flood_map)

# 수정된 부분: st.folium_static -> folium_static
folium_static(flood_map, width=1200, height=700)

st.markdown("---")
st.subheader("구현 설명")
st.markdown(
    """
    이 앱은 다음 단계를 통해 해수면 상승을 시뮬레이션합니다:

    1.  **가상 고도 데이터 생성**: 전 세계의 대략적인 고도 데이터를 `numpy` 배열로 생성합니다. 실제 앱에서는 SRTM, ASTER GDEM과 같은 **Digital Elevation Model (DEM) 파일을 `rasterio` 라이브러리를 사용하여 로드**해야 합니다.
    2.  **해수면 상승 예측**: IPCC 보고서 등을 참고하여 미래 연도별 예상 해수면 상승량(센티미터)을 정의하고, 선택된 연도에 따라 보간합니다.
    3.  **침수 지역 마스킹**: 고도 데이터에서 예상 해수면 상승량보다 낮은 모든 지점을 '침수 지역'으로 마스킹합니다.
    4.  **지도 시각화**: `folium` 라이브러리를 사용하여 배경 지도를 생성하고, 침수 마스크를 투명한 파란색 오버레이 이미지로 변환하여 지도 위에 겹쳐 표시합니다.
    """
)
st.warning(
    """
    **실제 데이터를 사용하려면:**
    이 코드의 `generate_fake_dem_data` 함수를 실제 DEM 파일을 로드하는 코드로 대체해야 합니다.
    예를 들어, `.tif` 형식의 DEM 파일을 다운로드하여 `rasterio` 라이브러리로 읽어올 수 있습니다.
    하지만 전 세계 고해상도 DEM은 파일 크기가 매우 크고, Streamlit 앱에 배포하기 전에 전처리가 필요할 수 있습니다.
    작은 지역 단위로 먼저 시도해 보시는 것을 권장합니다.
    """
)