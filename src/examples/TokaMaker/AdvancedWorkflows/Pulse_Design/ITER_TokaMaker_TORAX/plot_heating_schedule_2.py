import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE (BIG FONT MODE)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'lines.linewidth': 2.0,
})

# ─────────────────────────────────────────────────────────────────────────────
# BASE SCHEDULES
# ─────────────────────────────────────────────────────────────────────────────
_BASE_RL_ECRH_POWERS_W = {
    0: 0.0,
    10: 0.0,
    50: 20.0e6,
    80: 20.0e6,
    520: 10.0e6,
    540: 5.0e6,
    560: 0.0,
    600: 0.0,
}

_BASE_RL_NBI_POWERS_W = {
    0: 0.0,
    79: 0.0,
    80: 33.0e6,
    520: 10.0e6,
    540: 7.0e6,
    560: 4.0e6,
    580: 2.0e6,
    600: 0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT DECISIONS
# ─────────────────────────────────────────────────────────────────────────────
agent_decisions = [
    {'decision_t': 80.0, 'knot_t': 100.0, 'ecrh_W': 15511136.0, 'nbi_W': 11716048.0},
{'decision_t': 100.0, 'knot_t': 120.0, 'ecrh_W': 12915730.0, 'nbi_W': 9108131.0},
{'decision_t': 120.0, 'knot_t': 140.0, 'ecrh_W': 11163807.0, 'nbi_W': 8159074.0},
{'decision_t': 140.0, 'knot_t': 160.0, 'ecrh_W': 9234359.0, 'nbi_W': 7252773.5},
{'decision_t': 160.0, 'knot_t': 180.0, 'ecrh_W': 8681418.0, 'nbi_W': 7126561.0},
{'decision_t': 180.0, 'knot_t': 200.0, 'ecrh_W': 8533120.0, 'nbi_W': 7753004.5},
{'decision_t': 200.0, 'knot_t': 220.0, 'ecrh_W': 8935236.0, 'nbi_W': 9464709.0},
{'decision_t': 220.0, 'knot_t': 240.0, 'ecrh_W': 8820467.0, 'nbi_W': 10229731.0},
{'decision_t': 240.0, 'knot_t': 260.0, 'ecrh_W': 7449112.0, 'nbi_W': 11138896.0},
{'decision_t': 260.0, 'knot_t': 280.0, 'ecrh_W': 6517694.0, 'nbi_W': 11244717.0},
{'decision_t': 280.0, 'knot_t': 300.0, 'ecrh_W': 4684749.0, 'nbi_W': 11544795.0},
{'decision_t': 300.0, 'knot_t': 320.0, 'ecrh_W': 3253462.25, 'nbi_W': 11197866.0},
{'decision_t': 320.0, 'knot_t': 340.0, 'ecrh_W': 2778619.75, 'nbi_W': 11446301.0},
{'decision_t': 340.0, 'knot_t': 360.0, 'ecrh_W': 2325089.75, 'nbi_W': 11863594.0},
{'decision_t': 360.0, 'knot_t': 380.0, 'ecrh_W': 2292788.75, 'nbi_W': 12003206.0},
{'decision_t': 380.0, 'knot_t': 400.0, 'ecrh_W': 2309985.75, 'nbi_W': 12200740.0},
{'decision_t': 400.0, 'knot_t': 420.0, 'ecrh_W': 2053837.625, 'nbi_W': 12537725.0},
{'decision_t': 420.0, 'knot_t': 440.0, 'ecrh_W': 2108873.75, 'nbi_W': 12278580.0},
{'decision_t': 440.0, 'knot_t': 460.0, 'ecrh_W': 2258911.75, 'nbi_W': 12234252.0},
{'decision_t': 460.0, 'knot_t': 480.0, 'ecrh_W': 1711160.375, 'nbi_W': 11209463.0},
{'decision_t': 480.0, 'knot_t': 500.0, 'ecrh_W': 1762944.5, 'nbi_W': 12329789.0}

]

# ─────────────────────────────────────────────────────────────────────────────
def schedule_to_arrays(d):
    t = np.array(sorted(d.keys()), dtype=float)
    v = np.array([d[k] for k in t]) / 1e6
    return t, v

def interp(d, tgrid):
    t, v = schedule_to_arrays(d)
    return np.interp(tgrid, t, v)

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────────────────
t = np.linspace(0, 600, 2000)

base_ecrh = interp(_BASE_RL_ECRH_POWERS_W, t)
base_nbi  = interp(_BASE_RL_NBI_POWERS_W, t)

mask = (t >= 80) & (t <= 520)

base_ecrh[mask] = np.interp(t[mask], [80, 520], [20.0, 10.0])
base_nbi[mask]  = np.interp(t[mask], [80, 520], [33.0, 10.0])

# ─────────────────────────────────────────────────────────────────────────────
# AGENT (connected line)
# ─────────────────────────────────────────────────────────────────────────────
agent_t = np.array([d['knot_t'] for d in agent_decisions])
agent_ecrh = np.array([d['ecrh_W'] for d in agent_decisions]) / 1e6
agent_nbi  = np.array([d['nbi_W'] for d in agent_decisions]) / 1e6

agent_t_full = np.concatenate([[80], agent_t, [520]])
agent_ecrh_full = np.concatenate([[20.0], agent_ecrh, [10.0]])
agent_nbi_full  = np.concatenate([[33.0], agent_nbi, [10.0]])

idx = np.argsort(agent_t_full)
agent_t_full = agent_t_full[idx]
agent_ecrh_full = agent_ecrh_full[idx]
agent_nbi_full = agent_nbi_full[idx]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.axvspan(0, 80, color='#ffcccc', alpha=0.4)
ax.axvspan(80, 520, color='#ccffcc', alpha=0.4)
ax.axvspan(520, 600, color='#ffcccc', alpha=0.4)

# baseline
ax.plot(t, base_ecrh, '--', color='#4477cc', label='ECRH baseline')
ax.plot(t, base_nbi,  '--', color='#ee8833', label='NBI baseline')

# agent
ax.plot(agent_t_full, agent_ecrh_full,
        '-o', color='#4477cc', lw=3, ms=7, label='ECRH agent')

ax.plot(agent_t_full, agent_nbi_full,
        '-o', color='#ee8833', lw=3, ms=7, label='NBI agent')

# labels
ax.set_xlabel('Time (s)')
ax.set_ylabel('Power (MW)')
ax.set_title('Baseline vs Agent ECRH/NBI Schedule')

ax.set_xlim(0, 600)
ax.grid(True, alpha=0.3)

# BIG LEGEND
fixed_patch = mpatches.Patch(color='#ffcccc', alpha=0.6, label='Fixed regions')
agent_patch = mpatches.Patch(color='#ccffcc', alpha=0.6, label='Agent-controlled')

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles + [fixed_patch, agent_patch],
    loc='upper right',
    frameon=True,
    fontsize=16,
    title='Legend',
    title_fontsize=16
)

plt.tight_layout()
plt.show()
