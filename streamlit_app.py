import streamlit as st
import geopandas as gpd
import pandas as pd
import pydeck as pdk
from shapely.ops import unary_union, transform as shp_transform
from shapely.geometry import Polygon, MultiPolygon
import shapely.geometry as geom
import numpy as np
import pyproj
import altair as alt

st.set_page_config(layout="wide", page_title="Sea Level Rise Simulator")

st.title("해수면 상승 시뮬레이터 — Streamlit 데모 (수정본)")

# -------------------- 설명
st.markdown(
    """
    이 수정본은 Streamlit의 캐시 에러(`UnhashableParamError`)를 해결하기 위해
    다음 원칙을 적용했다:

    1) Streamlit 캐시(`@st.cache_data`)는 함수 인자로 들어오는 객체를 해시하려고 시도한다.
       GeoDataFrame이나 Shapely geometry 같은 객체는 기본적으로 해시 불가라서 에러가 발생한다.
    2) 해결책은 "(A) 해시 불가 인자를 캐시 함수 인자로 넘기지 않기" 또는
       "(B) 캐시할 때 타입별 해시 함수를 제공(hash_funcs)" 이다.

    이 파일은 안정성과 단순함을 위해 (A)를 택해, GeoDataFrame/Geometry를 인자로 받는
    함수들에는 캐시를 적용하지 않았다. 대신, `load_world()`는 캐시한다(파일/네트워크 I/O를 줄이기 위해).
    """
)

# -------------------- 사용자 입력
years = st.sidebar.multiselect("연도 선택 (시뮬레이션)", [2030, 2040, 2050, 2060, 2070], default=[2030, 2040, 2050])
if not years:
    st.sidebar.error("적어도 한 개의 연도를 선택해라")

region = st.sidebar.selectbox("지역 선택", ["World", "Asia", "North America", "South America", "Africa", "South Korea"]) 

# map of example sea level projections (m) — 사용자는 필요시 조정 가능
default_slr = {2030: 0.3, 2040: 0.6, 2050: 1.0, 2060: 1.6, 2070: 2.2}

# 사용자가 값 조정 가능
st.sidebar.markdown("---")
st.sidebar.markdown("### 연도별 해수면 상승(m) (편의 입력)")
for y in sorted(default_slr.keys()):
    default_slr[y] = st.sidebar.number_input(f"{y} (m)", value=float(default_slr[y]), step=0.1, format="%.2f")

# -------------------- 데이터 준비 (Natural Earth)
@st.cache_data
def load_world():
    """Load Natural Earth countries as GeoDataFrame and normalize columns."""
    try:
        path = gpd.datasets.get_path('naturalearth_lowres')
        world = gpd.read_file(path)
    except Exception:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)

    world.columns = [c.lower() for c in world.columns]

    if 'name' not in world.columns:
        for alt in ['admin', 'sovereignt', 'name_long', 'name_en', 'name_ascii']:
            if alt in world.columns:
                world = world.rename(columns={alt: 'name'})
                break

    if 'continent' not in world.columns:
        for alt in ['region_un', 'region_wb', 'continent_en']:
            if alt in world.columns:
                world = world.rename(columns={alt: 'continent'})
                break

    return world

world = load_world()

# -------------------- 지역 기하 도출 (캐시하지 않음)
def get_region_geom(world_gdf, region_key):
    if region_key == 'World':
        return world_gdf.unary_union
    if region_key == 'South Korea':
        if 'name' in world_gdf.columns:
            korea_mask = world_gdf['name'].str.lower().str.contains('korea', na=False)
            korea_df = world_gdf[korea_mask]
            if not korea_df.empty:
                south = korea_df[korea_df['name'].str.lower().str.contains('south', na=False)]
                if not south.empty:
                    return south.unary_union
                return korea_df.unary_union
    if 'continent' in world_gdf.columns:
        match = world_gdf[world_gdf['continent'].str.lower() == region_key.lower()]
        if not match.empty:
            return match.unary_union
    return world_gdf.unary_union

# -------------------- 해안선(조합) 생성 — 캐시 제거
# (이 함수를 캐시하면 GeoDataFrame 인자 때문에 UnhashableParamError가 난다.)
def build_coastline_union(world_gdf):
    boundaries = world_gdf.geometry.boundary
    unioned = unary_union(boundaries.values)
    return unioned

coastlines = build_coastline_union(world)

# -------------------- 좌표 변환 유틸
wgs84 = pyproj.CRS('EPSG:4326')
metric_crs = pyproj.CRS('EPSG:3857')
project_to_metric = pyproj.Transformer.from_crs(wgs84, metric_crs, always_xy=True).transform
project_to_wgs = pyproj.Transformer.from_crs(metric_crs, wgs84, always_xy=True).transform

# 작은 래퍼
def to_metric(geom_obj):
    return shp_transform(project_to_metric, geom_obj)

def to_wgs(geom_obj):
    return shp_transform(project_to_wgs, geom_obj)

# -------------------- 시뮬레이션 함수 — 캐시 제거
# Geo/Geometry 객체를 인자로 받으면 Streamlit 캐시가 실패하므로 캐시를 적용하지 않는다.
def simulate_inundation_for_years(years_list, slr_map, coastlines_geom, region_mask, meter_per_m):
    results = {}
    coast_metric = shp_transform(project_to_metric, coastlines_geom)
    region_metric = shp_transform(project_to_metric, region_mask)

    for y in years_list:
        slr_m = slr_map.get(y, 0.0)
        buffer_dist = slr_m * meter_per_m
        inundation_metric = coast_metric.buffer(buffer_dist)
        inundation_region = inundation_metric.intersection(region_metric)
        results[y] = shp_transform(project_to_wgs, inundation_region)
    return results

# 시뮬레이션 파라미터
METER_PER_MSLR = st.sidebar.slider("해수면 1m 상승당 내륙 침수 거리(m)", min_value=50, max_value=2000, value=400, step=10)

selected_region_geom = get_region_geom(world, region)
inundations = simulate_inundation_for_years(years, default_slr, coastlines, selected_region_geom, METER_PER_MSLR)

# -------------------- pydeck 변환 유틸
def geom_to_pydeck_polygons(geom_obj):
    polys = []
    if geom_obj is None or getattr(geom_obj, 'is_empty', True):
        return polys

    # 지원: Polygon, MultiPolygon, GeometryCollection
    geoms = []
    if isinstance(geom_obj, (Polygon, MultiPolygon)):
        if isinstance(geom_obj, MultiPolygon):
            geoms = list(geom_obj.geoms)
        else:
            geoms = [geom_obj]
    else:
        # try to iterate
        try:
            geoms = list(geom_obj.geoms)
        except Exception:
            geoms = []

    for p in geoms:
        try:
            exterior_coords = [[lon, lat] for lon, lat in p.exterior.coords]
            polys.append({"path": exterior_coords})
        except Exception:
            continue
    return polys

# 요약 테이블 생성
summary_rows = []
for y, poly in inundations.items():
    area_km2 = 0.0
    if poly is not None and not getattr(poly, 'is_empty', True):
        poly_metric = shp_transform(project_to_metric, poly)
        area_km2 = poly_metric.area / 1e6
    summary_rows.append({"year": y, "inundation_km2": area_km2})

summary_df = pd.DataFrame(summary_rows).sort_values('year')

# 지역 면적 대비 백분율
region_area_km2 = None
try:
    region_metric = shp_transform(project_to_metric, selected_region_geom)
    region_area_km2 = region_metric.area / 1e6
except Exception:
    region_area_km2 = None

if region_area_km2 and region_area_km2 > 0:
    summary_df['pct_of_region'] = summary_df['inundation_km2'] / region_area_km2 * 100
else:
    summary_df['pct_of_region'] = None

# 지도뷰 세팅
VIEW_PRESETS = {
    'World': {'latitude': 20, 'longitude': 0, 'zoom': 1},
    'Asia': {'latitude': 25, 'longitude': 100, 'zoom': 2.2},
    'North America': {'latitude': 45, 'longitude': -100, 'zoom': 2.2},
    'South America': {'latitude': -15, 'longitude': -60, 'zoom': 2.5},
    'Africa': {'latitude': 5, 'longitude': 20, 'zoom': 2.2},
    'South Korea': {'latitude': 36.5, 'longitude': 127.8, 'zoom': 5.5}
}

view = VIEW_PRESETS.get(region, VIEW_PRESETS['World'])

# pydeck 레이어 생성
country_layer = pdk.Layer(
    "GeoJsonLayer",
    data=world.__geo_interface__,
    stroked=True,
    filled=True,
    get_fill_color=[200, 200, 200, 50],
    get_line_color=[50, 50, 50],
    pickable=False,
)

inundation_layers = []
color_scale = [[255, 200, 200, 150], [200, 0, 0, 180], [180, 0, 80, 200], [120, 0, 120, 200], [80, 0, 200, 200]]
for i, (y, poly) in enumerate(sorted(inundations.items())):
    polygons = geom_to_pydeck_polygons(poly)
    if not polygons:
        continue
    layer = pdk.Layer(
        "PolygonLayer",
        data=polygons,
        get_polygon="path",
        pickable=True,
        extruded=False,
        stroked=False,
        opacity=0.6,
        get_fill_color=color_scale[i % len(color_scale)],
    )
    inundation_layers.append(layer)

r = pdk.Deck(
    layers=[country_layer] + inundation_layers,
    initial_view_state=pdk.ViewState(latitude=view['latitude'], longitude=view['longitude'], zoom=view['zoom'], pitch=0),
    tooltip={"html": "<b>Inundation layer</b>", "style": {"color": "white"}}
)

st.subheader("지도: 선택한 연도의 침수 영역(데모)")
st.pydeck_chart(r)

# 연도별 선 그래프
st.subheader("연도별 침수 면적(데모)")
st.line_chart(summary_df.set_index('year')['inundation_km2'])

# 히트맵: 지역별 비교
st.subheader("지역별 비교 히트맵 (데모)")
regions_to_compare = ['Asia', 'North America', 'South America', 'Africa', 'South Korea']
heat_rows = []
for reg in regions_to_compare:
    reg_geom = get_region_geom(world, reg)
    sim = simulate_inundation_for_years(years, default_slr, coastlines, reg_geom, METER_PER_MSLR)
    for y, p in sim.items():
        area_km2 = 0.0
        if p is not None and not getattr(p, 'is_empty', True):
            area_km2 = shp_transform(project_to_metric, p).area / 1e6
        heat_rows.append({'region': reg, 'year': y, 'inundation_km2': area_km2})

heat_df = pd.DataFrame(heat_rows)
if not heat_df.empty:
    pivot = heat_df.pivot(index='region', columns='year', values='inundation_km2').fillna(0)
    heat = alt.Chart(pivot.reset_index().melt(id_vars=['region'], var_name='year', value_name='km2')).mark_rect().encode(
        x='year:O',
        y='region:O',
        color='km2:Q'
    ).properties(width=700, height=200)
    st.altair_chart(heat)
else:
    st.write("히트맵용 데이터가 없습니다.")

# 요약 및 실행 방법
st.markdown("---")
st.write("**요약**: 캐시 에러는 GeoDataFrame/Geometry를 Streamlit 캐시 함수의 인자로 넘기면 발생한다. 이 코드는 해당 인자들에 캐시를 적용하지 않음으로써 문제를 회피한다.")
st.markdown("""### 실행 방법
1) 필요한 패키지 설치: `pip install streamlit geopandas pydeck shapely altair`
2) 실행: `streamlit run streamlit_sea_level_app.py`
""")
