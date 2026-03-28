import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Circle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  STYLE
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': '#f8f9fa',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#dfe6e9',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.8,
})


# ─────────────────────────────────────────────
#  FUZZY CONTROLLER
# ─────────────────────────────────────────────
class FuzzyTrafficController:
    def __init__(self):
        self.density_mf = {
            'Low': self._trapezoidal(0, 0, 20, 35),
            'Medium': self._triangular(25, 40, 55),
            'High': self._trapezoidal(45, 55, 60, 60),
        }
        self.waiting_mf = {
            'Short': self._trapezoidal(0, 0, 20, 40),
            'Medium': self._triangular(25, 45, 65),
            'Long': self._trapezoidal(50, 70, 90, 90),
        }
        self.green_mf = {
            'Short': self._triangular(5, 10, 15),
            'Medium': self._triangular(12, 20, 28),
            'Long': self._triangular(25, 35, 45),
            'Very Long': self._trapezoidal(40, 50, 60, 60),
        }

        self.rules = [
            {'density': 'Low', 'waiting': 'Short', 'output': 'Short'},
            {'density': 'Low', 'waiting': 'Medium', 'output': 'Medium'},
            {'density': 'Low', 'waiting': 'Long', 'output': 'Long'},
            {'density': 'Medium', 'waiting': 'Short', 'output': 'Medium'},
            {'density': 'Medium', 'waiting': 'Medium', 'output': 'Long'},
            {'density': 'Medium', 'waiting': 'Long', 'output': 'Very Long'},
            {'density': 'High', 'waiting': 'Short', 'output': 'Long'},
            {'density': 'High', 'waiting': 'Medium', 'output': 'Very Long'},
            {'density': 'High', 'waiting': 'Long', 'output': 'Very Long'},
        ]

    def _triangular(self, a, b, c):
        return lambda x: max(min((x-a)/(b-a), (c-x)/(c-b)), 0)

    def _trapezoidal(self, a, b, c, d):
        return lambda x: max(min((x-a)/(b-a) if b!=a else 1,
                                 1,
                                 (d-x)/(d-c) if d!=c else 1), 0)

    def fuzzify(self, value, mf_dict):
        return {k: mf(value) for k, mf in mf_dict.items()}

    def evaluate_rules(self, d, w):
        rules = []
        for r in self.rules:
            s = min(d[r['density']], w[r['waiting']])
            if s > 0:
                rules.append({
                    **r,
                    'density_label': r['density'],
                    'waiting_label': r['waiting'],
                    'strength': s,
                })
        return rules

    def defuzzify(self, rules):
        if not rules: return 20
        num = den = 0
        for x in np.arange(0, 60, 0.5):
            # Mamdani aggregation: clip each consequent by rule strength,
            # then take the maximum across fired rules at this x.
            agg_m = max(min(r['strength'], self.green_mf[r['output']](x)) for r in rules)
            num += x * agg_m
            den += agg_m
        return num/den if den else 20

    def calculate_green_time(self, d, w):
        df = self.fuzzify(d, self.density_mf)
        wf = self.fuzzify(w, self.waiting_mf)
        rules = self.evaluate_rules(df, wf)
        g = self.defuzzify(rules)
        return g, df, wf, rules


# ─────────────────────────────────────────────
#  COLOUR PALETTES  (plain hex — no alpha hacks)
# ─────────────────────────────────────────────
DENSITY_COLORS = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
WAITING_COLORS = {'Short': '#27ae60', 'Medium': '#e67e22', 'Long': '#c0392b'}
OUTPUT_COLORS  = {
    'Short': '#74b9ff', 'Medium': '#0984e3',
    'Long':  '#6c5ce7', 'Very Long': '#636e72',
}
FILL_ALPHA = 0.18   # matplotlib handles alpha natively — no tricks needed


# ─────────────────────────────────────────────
#  HELPER — shared MF plot logic
# ─────────────────────────────────────────────
def _draw_mf_axes(ax, mf_dict, color_dict, x_arr, current_val, xlabel, title):
    """Draw membership functions with fills, current-value line & dot."""
    handles = []
    for name, mf in mf_dict.items():
        c = color_dict[name]
        y = np.array([mf(xi) for xi in x_arr])
        ax.fill_between(x_arr, y, alpha=FILL_ALPHA, color=c)
        line, = ax.plot(x_arr, y, color=c, linewidth=2.4, label=name)
        handles.append(line)
        # dot at current value
        deg = mf(current_val)
        ax.plot(current_val, deg, 'o', color=c, markersize=8,
                markeredgecolor='white', markeredgewidth=1.8, zorder=5)
        if deg > 0.02:
            ax.annotate(f'μ={deg:.2f}', xy=(current_val, deg),
                        xytext=(6, 4), textcoords='offset points',
                        fontsize=8.5, color=c, fontweight='bold')

    # vertical guide line
    ax.axvline(current_val, color='#2d3436', linewidth=1.4,
               linestyle=':', alpha=0.7, zorder=4)
    ax.text(current_val, 1.06, f'{current_val:.1f}',
            ha='center', va='bottom', fontsize=9.5,
            color='#2d3436', fontweight='bold',
            transform=ax.get_xaxis_transform())

    ax.set_xlim(x_arr[0], x_arr[-1])
    ax.set_ylim(0, 1.18)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Membership Degree  μ(x)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(handles=handles, loc='upper right', fontsize=9,
              framealpha=0.85, edgecolor='#dfe6e9')
    ax.tick_params(labelsize=9)


# ─────────────────────────────────────────────
#  MEMBERSHIP FUNCTION FIGURES
# ─────────────────────────────────────────────
def plot_density_mf(controller, density_val):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    _draw_mf_axes(ax, controller.density_mf, DENSITY_COLORS,
                  np.linspace(0, 60, 300), density_val,
                  'Vehicles per minute', 'Input 1 — Traffic Density')
    fig.tight_layout()
    return fig


def plot_waiting_mf(controller, waiting_val):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    _draw_mf_axes(ax, controller.waiting_mf, WAITING_COLORS,
                  np.linspace(0, 90, 300), waiting_val,
                  'Seconds', 'Input 2 — Waiting Time')
    fig.tight_layout()
    return fig


def plot_output_mf(controller, green_time):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.linspace(0, 60, 300)
    handles = []
    for name, mf in controller.green_mf.items():
        c = OUTPUT_COLORS[name]
        y = np.array([mf(xi) for xi in x])
        ax.fill_between(x, y, alpha=FILL_ALPHA, color=c)
        line, = ax.plot(x, y, color=c, linewidth=2.4, label=name)
        handles.append(line)

    # COG result line
    ax.axvline(green_time, color='#d63031', linewidth=2.2,
               linestyle='--', zorder=5, label=f'COG = {green_time:.1f} s')
    ax.text(green_time, 1.06, f'COG = {green_time:.1f} s',
            ha='center', va='bottom', fontsize=10,
            color='#d63031', fontweight='bold',
            transform=ax.get_xaxis_transform())

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel('Green Light Duration (seconds)', fontsize=10)
    ax.set_ylabel('Membership Degree  μ(x)', fontsize=10)
    ax.set_title('Output — Green Light Duration', fontsize=12,
                 fontweight='bold', pad=10)
    ax.legend(handles=handles, loc='upper right', fontsize=9,
              framealpha=0.85, edgecolor='#dfe6e9')
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  RULE ACTIVATION BAR CHART
# ─────────────────────────────────────────────
def plot_rule_activation(rules):
    labels    = [f"D:{r['density_label']}\nW:{r['waiting_label']}\n→{r['output']}"
                 for r in rules]
    strengths = [r['strength'] for r in rules]
    bar_colors = ['#0984e3' if s >= 0.5 else '#74b9ff' for s in strengths]

    fig, ax = plt.subplots(figsize=(max(6, len(rules) * 1.4), 4))
    bars = ax.bar(labels, strengths, color=bar_colors,
                  edgecolor='white', linewidth=1.2, width=0.55)
    for bar, val in zip(bars, strengths):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color='#2d3436')

    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Firing Strength', fontsize=10)
    ax.set_title('Rule Firing Strengths', fontsize=12,
                 fontweight='bold', pad=10)
    ax.tick_params(axis='x', labelsize=8.5)
    ax.tick_params(axis='y', labelsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  RULE HEATMAP  (matplotlib imshow)
# ─────────────────────────────────────────────
def plot_rule_heatmap():
    density_cats = ['Low', 'Medium', 'High']
    waiting_cats = ['Short', 'Medium', 'Long']
    output_map = {
        ('Low','Short'): 'Short',     ('Low','Medium'): 'Medium',    ('Low','Long'): 'Long',
        ('Medium','Short'): 'Medium', ('Medium','Medium'): 'Long',   ('Medium','Long'): 'Very Long',
        ('High','Short'): 'Long',     ('High','Medium'): 'Very Long',('High','Long'): 'Very Long',
    }
    val_map  = {'Short': 1, 'Medium': 2, 'Long': 3, 'Very Long': 4}
    z    = np.array([[val_map[output_map[(d, w)]] for w in waiting_cats] for d in density_cats], dtype=float)
    text = [[output_map[(d, w)] for w in waiting_cats] for d in density_cats]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'traffic', ['#74b9ff', '#0984e3', '#6c5ce7', '#2d3436'])
    im = ax.imshow(z, cmap=cmap, vmin=0.5, vmax=4.5, aspect='auto')

    ax.set_xticks(range(len(waiting_cats)))
    ax.set_xticklabels(waiting_cats, fontsize=10)
    ax.set_yticks(range(len(density_cats)))
    ax.set_yticklabels(density_cats, fontsize=10)
    ax.set_xlabel('Waiting Time', fontsize=10)
    ax.set_ylabel('Traffic Density', fontsize=10)
    ax.set_title('Rule Output Matrix', fontsize=12, fontweight='bold', pad=10)
    ax.grid(False)

    for i in range(len(density_cats)):
        for j in range(len(waiting_cats)):
            ax.text(j, i, text[i][j], ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if z[i, j] >= 3 else '#2d3436')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  TRAFFIC LIGHT  (matplotlib patches)
# ─────────────────────────────────────────────
def create_traffic_light(green_time):
    if green_time > 30:   active, label = 'green',  'EXTENDED GREEN'
    elif green_time > 15: active, label = 'yellow', 'NORMAL GREEN'
    else:                 active, label = 'red',    'SHORT GREEN'

    bright = {'red': '#ff4757', 'yellow': '#ffa502', 'green': '#2ed573'}
    dim    = {'red': '#3d0000', 'yellow': '#3d2600', 'green': '#003d10'}

    fig, ax = plt.subplots(figsize=(2.2, 4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 4.2)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # housing
    housing = FancyBboxPatch((0.1, 0.05), 0.8, 3.4,
                              boxstyle='round,pad=0.04',
                              facecolor='#2d3436', edgecolor='#636e72', linewidth=1.5)
    ax.add_patch(housing)

    # lights
    for light, yc in zip(['red', 'yellow', 'green'], [3.0, 2.0, 1.0]):
        is_on = (light == active)
        color = bright[light] if is_on else dim[light]
        circle = Circle((0.5, yc), 0.3, facecolor=color,
                        edgecolor='#636e72', linewidth=0.8,
                        alpha=1.0 if is_on else 0.5, zorder=3)
        ax.add_patch(circle)
        if is_on:
            # glare highlight
            glare = Circle((0.38, yc + 0.12), 0.1,
                           facecolor='white', alpha=0.25, zorder=4)
            ax.add_patch(glare)

    ax.text(0.5, 3.75, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#2d3436')
    ax.text(0.5, 3.55, f'{green_time:.1f} s', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#d63031')

    fig.tight_layout(pad=0.2)
    return fig


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────

st.set_page_config(layout="wide")

if 'density_slider' not in st.session_state:
    st.session_state['density_slider'] = 25.0
if 'waiting_slider' not in st.session_state:
    st.session_state['waiting_slider'] = 30.0

# Apply preset updates before sliders are instantiated.
if 'pending_density_slider' in st.session_state:
    st.session_state['density_slider'] = float(
        st.session_state.pop('pending_density_slider')
    )
if 'pending_waiting_slider' in st.session_state:
    st.session_state['waiting_slider'] = float(
        st.session_state.pop('pending_waiting_slider')
    )

if 'density_val' not in st.session_state:
    st.session_state['density_val'] = st.session_state['density_slider']
if 'waiting_val' not in st.session_state:
    st.session_state['waiting_val'] = st.session_state['waiting_slider']


st.title('🚦 Fuzzy Logic Traffic Light Controller')
st.caption('Mamdani fuzzy inference system for adaptive traffic signal control.')

controller = FuzzyTrafficController()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '📖 Fuzzy Rules',
    '🚦 Live Controller',
    '📈 Membership Functions',
    '🔍 Rule Activation',
    '🎮 Scenarios',
])

# ══════════════════════════════════════════
#  TAB 1 — FUZZY RULES
# ══════════════════════════════════════════
with tab1:
    st.header('Fuzzy Rule System — Design & Explanation')

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader('What is Fuzzy Logic?')
        st.markdown("""
Conventional controllers use hard thresholds — if vehicles > 30, switch to green.
This causes abrupt, unnatural transitions.

**Fuzzy logic** lets variables belong *partially* to multiple categories simultaneously,
mirroring human judgment:

> *"Traffic is 60 % medium and 40 % high — so grant a fairly long green."*

---
### Three-Step Mamdani Inference

**① Fuzzification**
Crisp sensor values are converted to membership degrees.
A density of 32 veh/min might register as *0.8 High* and *0.2 Medium* at the same time.

**② Rule Evaluation**
Each of the 9 IF–THEN rules fires with a strength equal to the
**minimum** of its two input membership degrees (AND = min).

**③ Defuzzification**
All fired, clipped output sets are aggregated.
The **Centre of Gravity (COG)** of the union area gives the final green-light duration.
""")
    with c2:
        st.subheader('Linguistic Variables')
        st.markdown("""
| Variable | Range | Sets |
|---|---|---|
| **Traffic Density** | 0–60 veh/min | Low · Medium · High |
| **Waiting Time** | 0–90 s | Short · Medium · Long |
| **Green Duration** | 0–60 s | Short · Medium · Long · Very Long |
""")
        st.subheader('Membership Function Types')
        st.markdown("""
| Shape | Used for | Why |
|---|---|---|
| **Trapezoidal** | Boundary sets (Low, High, Short, Long) | Plateau at 1.0 for extremes |
| **Triangular** | Central sets (Medium) | Sharp, well-defined peak |
""")

    st.divider()
    st.subheader('📋 Rule Base — 9 Rules')
    col_heat, col_list = st.columns([1, 1])
    with col_heat:
        st.pyplot(plot_rule_heatmap(), width='stretch')
        st.caption('Darker = longer green. Each cell = one fuzzy rule.')
    with col_list:
        rules_df = pd.DataFrame([
            {'#': i+1,
             'IF Density is': r['density'],
             'AND Waiting is': r['waiting'],
             'THEN Green =': r['output']}
            for i, r in enumerate(controller.rules)
        ])
        st.dataframe(rules_df, hide_index=True,
                     width='stretch', height=342)

    st.divider()
    st.subheader('🧠 Design Rationale')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success('**Low density + Short wait → Short green**\n\n'
                   'No other road should wait when traffic is sparse.')
    with c2:
        st.warning('**Medium density + Long wait → Very Long green**\n\n'
                   'Vehicles have queued for a while — priority must rise.')
    with c3:
        st.error('**High density + any wait → Long / Very Long**\n\n'
                 'Heavy flow always needs extended green to drain the queue.')

# ══════════════════════════════════════════
#  TAB 2 — LIVE CONTROLLER
# ══════════════════════════════════════════
with tab2:
    st.header('Live Controller')

    col_d, col_w = st.columns(2)
    with col_d:
        density = st.slider(
            '🚗 Traffic Density (veh/min)',
            min_value=0.0, max_value=60.0, step=0.5,
            # value=float(st.session_state['density_val']),
            key='density_slider',
        )
        st.session_state['density_val'] = density
    with col_w:
        waiting = st.slider(
            '⏱️ Waiting Time (seconds)',
            min_value=0.0, max_value=90.0, step=0.5,
            # value=float(st.session_state['waiting_val']),
            key='waiting_slider',
        )
        st.session_state['waiting_val'] = waiting

    green_time, d_fuzz, w_fuzz, active_rules = \
        controller.calculate_green_time(density, waiting)

    st.divider()
    col_light, col_metrics = st.columns([1, 2])
    with col_light:
        st.pyplot(create_traffic_light(green_time), width='content')
    with col_metrics:
        st.subheader('📊 Fuzzy Membership Values')
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric('Density — Low',    f"{d_fuzz['Low']:.2f}")
            st.metric('Density — Medium', f"{d_fuzz['Medium']:.2f}")
            st.metric('Density — High',   f"{d_fuzz['High']:.2f}")
        with m2:
            st.metric('Waiting — Short',  f"{w_fuzz['Short']:.2f}")
            st.metric('Waiting — Medium', f"{w_fuzz['Medium']:.2f}")
            st.metric('Waiting — Long',   f"{w_fuzz['Long']:.2f}")
        with m3:
            st.metric('🟢 Green Time',  f'{green_time:.1f} s')
            delta_txt = 'above avg' if green_time > 20 else 'below avg'
            st.metric('vs 20 s avg',   f'{green_time-20:+.1f} s', delta_txt)
            st.metric('Rules fired',   f'{len(active_rules)} / 9')

    st.divider()
    st.subheader('⚙️ Defuzzification — Centre of Gravity')
    st.caption(
        'Quick flow: fuzzify density/waiting → fire rules with min(μ_density, μ_waiting) '
        '→ aggregate clipped output sets → compute COG for final green time.'
    )
    st.latex(r'\text{COG} = \frac{\int x \cdot \mu(x)\, dx}{\int \mu(x)\, dx}')
    st.markdown(f'**Result: {green_time:.2f} s**')
    st.pyplot(plot_output_mf(controller, green_time),
              width='stretch')

# ══════════════════════════════════════════
#  TAB 3 — MEMBERSHIP FUNCTIONS
# ══════════════════════════════════════════
with tab3:
    st.header('Membership Function Graphs')
    st.caption('Shaded fills show each fuzzy set. '
               'The dotted vertical line marks your current input; '
               'the dot shows the membership degree at that point.')
    with st.expander('How to read these graphs (short notes)', expanded=True):
        st.markdown(
            '- Each curve maps a crisp input to a membership degree $\\mu(x)$ in $[0,1]$.\n'
            '- One input can belong to multiple sets at once (example: partly Medium and High).\n'
            '- For each rule, firing strength is $\\min(\\mu_{density}, \\mu_{waiting})$.\n'
            '- Fired rules clip output sets, then all clipped outputs are combined.\n'
            '- Final green time is the centroid (COG) of the combined output area.'
        )

    density    = st.session_state['density_val']
    waiting    = st.session_state['waiting_val']
    green_time, _, _, _ = controller.calculate_green_time(density, waiting)

    ca, cb = st.columns(2)
    with ca:
        st.pyplot(plot_density_mf(controller, density),
                  width='stretch')
    with cb:
        st.pyplot(plot_waiting_mf(controller, waiting),
                  width='stretch')

    st.pyplot(plot_output_mf(controller, green_time),
              width='stretch')
    st.info('💡 Adjust sliders in the **Live Controller** tab — '
            'this page updates automatically.')

# ══════════════════════════════════════════
#  TAB 4 — RULE ACTIVATION
# ══════════════════════════════════════════
with tab4:
    st.header('Rule Activation & Defuzzification')

    density    = st.session_state['density_val']
    waiting    = st.session_state['waiting_val']
    green_time, _, _, active_rules = \
        controller.calculate_green_time(density, waiting)

    if active_rules:
        st.pyplot(plot_rule_activation(active_rules),
                  width='stretch')
        st.subheader('Fired Rules Detail')
        for i, rule in enumerate(active_rules, 1):
            st.markdown(
                f"**Rule {i}:** IF density is **{rule['density_label']}** "
                f"AND waiting is **{rule['waiting_label']}** "
                f"→ Green = **{rule['output']}**"
            )
            st.progress(int(rule['strength'] * 100),
                        text=f"Firing strength: {rule['strength']:.3f}")
    else:
        st.warning('No rules fired with the current inputs.')

# ══════════════════════════════════════════
#  TAB 5 — SCENARIOS
# ══════════════════════════════════════════
with tab5:
    st.header('🎮 Preset Scenarios')
    st.markdown('Click **Load** to push values into the Live Controller '
                'and all chart tabs instantly.')

    scenarios = {
        '🏙️ Rush Hour':       {'density': 52.0, 'waiting': 72.0,
                                'desc': 'Peak traffic — dense flow and long queues.'},
        '🌙 Late Night':       {'density':  6.0, 'waiting':  8.0,
                                'desc': 'Near-empty roads. Short cycles save energy.'},
        '🏫 School Zone':      {'density': 28.0, 'waiting': 38.0,
                                'desc': 'Moderate traffic around school hours.'},
        '🚑 Emergency Flush':  {'density': 45.0, 'waiting': 80.0,
                                'desc': 'High density + long wait → maximum green.'},
        '🌤️ Weekend Morning':  {'density': 18.0, 'waiting': 20.0,
                                'desc': 'Light-to-medium, short waits.'},
    }

    cols = st.columns(len(scenarios))
    for col, (name, cfg) in zip(cols, scenarios.items()):
        with col:
            est_gt, _, _, _ = controller.calculate_green_time(
                cfg['density'], cfg['waiting'])
            st.markdown(f'**{name}**')
            st.caption(cfg['desc'])
            st.markdown(
                f"<small>🚗 {cfg['density']:.0f} veh/min &nbsp;|&nbsp; "
                f"⏱ {cfg['waiting']:.0f} s &nbsp;|&nbsp; "
                f"🟢 ~{est_gt:.0f} s</small>",
                unsafe_allow_html=True,
            )
            if st.button('Load', key=f'btn_{name}'):
                st.session_state['pending_density_slider'] = cfg['density']
                st.session_state['pending_waiting_slider'] = cfg['waiting']
                st.session_state['density_val'] = cfg['density']
                st.session_state['waiting_val'] = cfg['waiting']
                st.rerun()

    st.divider()
    st.subheader('Currently Loaded')
    d_now = st.session_state['density_val']
    w_now = st.session_state['waiting_val']
    gt_now, _, _, _ = controller.calculate_green_time(d_now, w_now)
    c1, c2, c3 = st.columns(3)
    c1.metric('Traffic Density', f'{d_now:.1f} veh/min')
    c2.metric('Waiting Time',    f'{w_now:.1f} s')
    c3.metric('Green Time',      f'{gt_now:.1f} s')