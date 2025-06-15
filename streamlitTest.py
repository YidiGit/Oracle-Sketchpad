import os
import base64
import open_clip
import torch
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ================================
# 环境修复：关闭文件监视，避免 torch.classes 报错
# ================================
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# ================================
# 常量与资源目录
# ================================
CODE_NAME = {
    "01":"狗","02":"猪","03":"鼠","04":"牛",
    "05":"虎","06":"兔","07":"龙","08":"蛇",
    "09":"马","10":"羊","11":"猴","12":"鸡"
}
ZODIAC_DETAILS = {
    "鼠":"机灵聪慧，善于适应环境。甲骨文中以尖嘴长尾示意其灵动。",
    "牛":"勤劳踏实，象征力量与付出。甲骨文字形突出犄角与强壮身躯。",
    "虎":"威猛勇敢，百兽之王。甲骨文以张口利爪和弯尾描绘其雄壮。",
    "兔":"温顺机敏，喜好夜行。甲骨文多以长耳与弯身示意。",
    "龙":"祥瑞神兽，变化万端。甲骨文以兽首蛇身展现神秘形态。",
    "蛇":"缠绕蜿蜒，灵动如虚线。甲骨文以曲线体现其柔韧身姿。",
    "马":"奔放自由，速度象征。甲骨文突出鬃毛与四足并行。",
    "羊":"温和善良，群居象征。甲骨文以卷角与毛绒线条刻画。",
    "猴":"机敏顽皮，善于攀爬。甲骨文常以弯肢与长尾描绘其灵活。",
    "鸡":"晨鸣勤奋，羽毛分明。甲骨文字形突出鸡冠与尾羽。",
    "狗":"忠诚守护，伴侣象征。甲骨文以卷尾与长身示意其忠诚。",
    "猪":"憨厚可爱，象征富足与平安。甲骨文以圆体与短尾描绘。"
}
ASSETS_DIR = "Data/assets"
DATASET_DIR = "Data/dataset"

# ================================
# 页面配置 & 全局 CSS
# ================================
st.set_page_config(
    page_title="博物馆甲骨文＆十二生肖互动",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="auto"
)
st.markdown("""
<style>
  .stApp { background: #c7c997 !important; color: #000 !important; }
  [data-testid="stSidebar"] { background-color: #EAE2B7 !important; color: #000!important; }
  [data-testid="stSidebar"] * { color: #000!important; }

  /* 普通按钮样式 */
  .stButton > button {
    background: linear-gradient(45deg,#85FFBD,#FFFB7D)!important;
    border:none!important;
    border-radius:50px!important;
    padding:10px 24px!important;
    font-size:16px!important;
    font-weight:600!important;
    color:#333!important;
    transition:transform .2s, box-shadow .2s!important;
  }
  .stButton > button:hover {
    transform: translateY(-2px)!important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.1)!important;
  }
  .stButton > button:active {
    transform: translateY(0)!important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2)!important;
  }

  /* 重点：针对 st.form_submit_button 的宿主容器 .stFormSubmitButton */
  .stFormSubmitButton > button {
    background: linear-gradient(45deg,#85FFBD,#FFFB7D)!important;
    border:none!important;
    border-radius:50px!important;
    padding:10px 24px!important;
    font-size:16px!important;
    font-weight:600!important;
    color:#333!important;
    transition:transform .2s, box-shadow .2s!important;
  }
  .stFormSubmitButton > button:hover {
    transform: translateY(-2px)!important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.1)!important;
  }
  .stFormSubmitButton > button:active {
    transform: translateY(0)!important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2)!important;
  }

  .canvas-toolbar { background: rgba(255,255,255,0.85)!important; }
  @media(max-width:600px){ .stCanvas>div{ width:100%!important; height:auto!important; } }
</style>
""", unsafe_allow_html=True)  

# ================================
# session_state: 页面状态
# ================================
if "page" not in st.session_state:
    st.session_state.page = "首页科普"

# ================================
# 侧边栏导航
# ================================
page = st.sidebar.radio(
    "🔖 导航",
    ("首页科普", "绘图体验", "反馈建议"),
    index=["首页科普","绘图体验","反馈建议"].index(st.session_state.page)
)
st.session_state.page = page

# ================================
# Base64 转码 & 模型加载
# ================================
def to_base64(path: str) -> str:
    data = open(path,"rb").read()
    ext = os.path.splitext(path)[1].lstrip(".")
    mime = f"image/{'jpeg' if ext in ['jpg','jpeg'] else ext}"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    return model.eval(), preprocess

@st.cache_data
def load_data():
    emb = np.load("model_make/animal_embeddings.npy")
    lbl = np.load("model_make/animal_labels.npy")
    names = sorted([d for d in os.listdir(DATASET_DIR) if not d.startswith(".")])
    return emb, lbl, names

model, preprocess = load_model()
embeddings, labels, class_names = load_data()

def predict_sketch(img_arr):
    img = Image.fromarray(img_arr.astype("uint8")).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        q = model.encode_image(tensor).squeeze().numpy()
    sims = (embeddings @ q)/(np.linalg.norm(embeddings,axis=1)*np.linalg.norm(q))
    scores = {cls:float(sims[labels==i].mean()) for i,cls in enumerate(class_names)}
    return max(scores, key=scores.get), scores

# ================================
# 各页面渲染
# ================================
def show_home():
    bg = os.path.join(ASSETS_DIR,"background2.png")
    if os.path.exists(bg):
        st.image(bg, use_container_width=True)
    else:
        st.warning("请在 Data/assets/ 下添加 background2.png")

    st.title("📜 甲骨文科普")
    st.write("""
甲骨文诞生于商代，用于占卜和记录，是研究中国早期文明的重要文字。  
- **发现**：1899年安阳殷墟  
- **用途**：祭祀、占卜、记录  
- **价值**：汉字起源的重要见证
    """)  

    st.header("🐭🐮🐯 十二生肖详解")
    col_text, col_img = st.columns([3,1], gap="large")
    with col_text:
        for code in class_names:
            animal = CODE_NAME[code]
            detail = ZODIAC_DETAILS[animal]
            uri = to_base64(f"{ASSETS_DIR}/Oracle_Bone/{code}.jpg")
            st.markdown(f"""
**{animal}**  
{detail}  
<img src="{uri}" style="width:90px;height:90px;object-fit:contain;" />
""", unsafe_allow_html=True)
    with col_img:
        zp = os.path.join(ASSETS_DIR,"zodiac.png")
        if os.path.exists(zp):
            st.image(zp, use_container_width=True, caption="Zodiac Overview")
        else:
            st.warning("请在 Data/assets/ 下添加 zodiac.png")

    st.markdown("---")
    st.markdown("""
## ✍️ 手绘识别体验  
在“绘图体验”页面，使用画布手绘甲骨文草图，点击【🚀 开始预测】即可查看 AI 识别结果。  
""", unsafe_allow_html=True)
    if st.button("🎨 去绘图体验"):
        st.session_state.page = "绘图体验"
        st.rerun()

def show_draw():
    st.title("✍️ 在线手绘识别")
    c1, c2 = st.columns([1,2], gap="large")
    with c1:
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)", stroke_width=3,
            stroke_color="#000", background_color="#fff",
            height=350, width=350, drawing_mode="freedraw",
            key="canvas"
        )
        if canvas.image_data is not None and st.button("🚀 开始预测"):
            st.session_state.pred = predict_sketch(canvas.image_data)
    with c2:
        if "pred" in st.session_state:
            best, scores = st.session_state.pred
            st.success(f"🔮 识别：{CODE_NAME[best]}")
            st.metric("置信度", f"{scores[best]*100:.1f}%")
            for cls, sc in sorted(scores.items(), key=lambda x:-x[1])[:3]:
                u1 = to_base64(f"{ASSETS_DIR}/Oracle_Bone/{cls}.jpg")
                u2 = to_base64(f"{ASSETS_DIR}/Real_Animals/{cls}.png")
                st.markdown(f"""
<div style="display:flex;gap:8px;align-items:center;margin:8px 0;">
  <img src="{u1}" style="width:80px;height:80px;border-radius:6px;object-fit:contain;"/>
  <img src="{u2}" style="width:80px;height:80px;border-radius:6px;object-fit:contain;"/>
  <span>{CODE_NAME[cls]} {sc*100:.1f}%</span>
</div>
""", unsafe_allow_html=True)
                st.progress(sc)
        else:
            st.info("请先在左侧画布绘制，然后点击“开始预测”")

    st.markdown("---")
    st.markdown("""
## 💬 留言 & 建议  
完成绘图识别后，欢迎您前往“反馈建议”页面，告诉我们您的体验和改进建议！
""", unsafe_allow_html=True)
    if st.button("💬 去反馈建议"):
        st.session_state.page = "反馈建议"
        st.rerun()

def show_feedback():
    st.title("💬 反馈建议")
    with st.form("fb"):
        st.text_input("您的称呼（选填）")
        st.slider("满意度评分",1,5,5)
        st.text_area("请留下您的建议")
        if st.form_submit_button("提交"):
            st.success("🙏 感谢您的反馈！")

# 根据页签渲染
if st.session_state.page == "首页科普":
    show_home()
elif st.session_state.page == "绘图体验":
    show_draw()
else:
    show_feedback()
