
#%%
import os
import numpy as np
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from mpl_toolkits import mplot3d

import seaborn as sns


%matplotlib qt

import farm

from farm.climate import Climate
from farm.soil import Soil
from farm.plant import Crop
from farm.model import CropModel


#%%

climate = Climate(
    alpha_r = 10.0,
    lambda_r = 0.3,
    t_seas = 180.,
    ET_max = 6.5,
    q_e = 1.5
)

soil = Soil('sand')

crop = Crop(
    KC_MAX = 1.2,
    LAI_MAX = 2.0,
    T_MAX = 4.0,
    Zr = 1000,
    sw_MPa = -1.5,
    s_star_MPa = -0.05,
    soil = soil,
)

model = CropModel(crop=crop, soil=soil, climate=climate)

model.run()

out = model.output()

out.ET.plot()

fig = plt.figure()
sns.scatterplot(out.s, out['T'], hue=out.dos)

#%%

clim_args = {
    'alpha_r': 10.0,
    'lambda_r': 0.3,
    't_seas': 180.,
    'ET_max': 6.5,
    'q_e': 1.5
}

soil_args = {
    'texture': 'sand'
}

crop_args = {
    'KC_MAX': 1.2,
    'LAI_MAX': 2.0,
    'T_MAX': 4.0,
    'Zr': 1000,
    'sw_MPa': -1.5,
    's_star_MPa': -0.05,
}


def run_sim(clim_args, soil_args, crop_args, return_mod=False):

    climate = Climate(**clim_args)
    soil = Soil(**soil_args)
    crop = Crop(soil=soil, **crop_args)
    
    model = CropModel(crop=crop, soil=soil, climate=climate)
    model.run()
    out = model.output()

    if return_mod:
        return model, out

    return out

def sims(clim_args, soil_args, crop_args, n_sims=100):

    dfs = []
    for i in range(n_sims):
        df = run_sim(clim_args, soil_args, crop_args)
        df['sim'] = i
        dfs.append(df)
    
    sims = pd.concat(dfs, ignore_index=True)

    # sims = pd.concat(
    #     [run_sim(clim_args, soil_args, crop_args) for _ in range(n_sims)], 
    #     ignore_index=True
    # )
    return sims

dfs = []

for s_star in [-0.1, -0.05, -0.01, -0.008]:
    crop_args['s_star_MPa'] = s_star
    df = sims(clim_args, soil_args, crop_args, n_sims=100)
    df['s_star_MPa'] = str(s_star)
    dfs.append(df)

full = pd.concat(dfs, ignore_index=True)


#%%

ets = []
trs = []
stars = []

for n in range(1000):
    ss = np.random.choice(np.arange(-0.3,-0.008,0.001),1)[0]
    crop_args['s_star_MPa'] = ss
    df = run_sim(clim_args, soil_args, crop_args)
    ets.append(df.ET.sum())
    trs.append(df['T'].sum())
    stars.append(ss)

df = pd.DataFrame({'ET': ets, 'Tr': trs, 's_star_MPa': stars})


#%%

fig = plt.figure()

sns.scatterplot(-1*df.s_star_MPa, df.ET, hue=df.Tr)

sns.regplot(-1*df.s_star_MPa, df.ET, logx=True, scatter=False, ci=None, color='k')
sns.regplot(-1*df.s_star_MPa, df['Tr'], logx=True, scatter=False, ci=None, color='k')



#%%

q_e = 1.5
s_star_MPa = -0.008
s_star_MPa = -0.3

sw_MPa = -1.5


climate = Climate(
    alpha_r = 10.0,
    lambda_r = 0.3,
    t_seas = 180.,
    ET_max = 6.5,
    q_e = q_e
)

soil = Soil('sand')

crop = Crop(
    KC_MAX = 1.2,
    LAI_MAX = 2.0,
    T_MAX = 4.0,
    Zr = 1000,
    sw_MPa = sw_MPa,
    s_star_MPa = s_star_MPa,
    soil = soil,
)



lai = 2.
sh = 0.42


s = np.arange(0,1,0.0056)


trans = [crop.calc_T(_s, LAI=lai) for _s in s]
evap = [climate.calc_E(_s, LAI=lai, sh=sh, ) for _s in s]
et = [crop.calc_T(_s, LAI=lai) + climate.calc_E(_s, LAI=lai, sh=sh) for _s in s ]


fig = plt.figure()
plt.plot(s, trans, '-')
plt.plot(s, evap, '--')
plt.plot(s, et, ':')






# %%
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
# mpl.rcParams['font.sans-serif'] = 'Source Sans Pro'
mpl.rcParams['font.size'] = 9.5
mpl.rcParams['axes.titlesize'] = 9.5

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# fig_fold = '/Users/brynmorgan/Library/Mobile Documents/com~apple~CloudDocs/Jobs/Postdoc Applications/Stanford Science Fellows/'
fig_fold = '/Users/brynmorgan/dev/pfp-23/'


#%%

theta = np.arange(-1, 2, 0.01)

def sigmoid(x, k, x0, L=1, b=0):
    y = L / ( 1 + np.exp( -k * ( x - x0 ) ) ) + b
    return (y)

et1 = sigmoid(theta, k=15, x0=0.3,)
et2 = sigmoid(theta, k=5, x0=1.,)

labs_dict = {
    0 : {
        'label' : r"$\theta_*$",
        'color' : '#1ebcd6',
        'x' : 0.525,
    },
    1 : {
        'label' : r"$\theta_*$",
        'color' : 'red',
        'x' : 1.725,
    },
}

# fig = plt.figure(figsize=(2.3,4.))
fig = plt.figure(figsize=(2.58,4.5))

ax1 = plt.subplot(2,1,1)

ax1.hlines(1, -1, 2, linestyle='--', color='k', linewidth=1.0)
ax1.vlines(
    x=[labs_dict[0]['x'], labs_dict[1]['x']],
    ymin=[-0.5,-0.5], 
    ymax=[sigmoid(labs_dict[0]['x'], k=15, x0=0.3,), sigmoid(labs_dict[1]['x'], k=5, x0=1.,)],
    linestyle='--', linewidth=1.0,
    color=[labs_dict[0]['color'], labs_dict[1]['color']],
)

ax1.plot(theta, et1, color=labs_dict[0]['color'])
ax1.plot(theta, et2, color=labs_dict[1]['color'])

ax1.set_xlabel(r"Soil moisture ($\theta$)") #, labelpad=5)
ax1.set_ylabel(r"$\frac{ET}{ET_{\mathrm{max}}}$", rotation=0) #, labelpad=10)

ax1.set_ylim(-0.05,1.1)
ax1.set_xlim(-0.25,2.1)

ax1.set_yticks([0,1])
ax1.set_xticks(
    [0, labs_dict[0]['x'], labs_dict[1]['x'], 2],
    [r"$\theta_{\mathrm{wp}}$", labs_dict[0]['label'], labs_dict[1]['label'], r"$\theta_{\mathrm{fc}}$"]
)

for i,(xticklab,xtick) in enumerate(zip(ax1.get_xticklabels()[1:3], ax1.get_xticklines()[1:3])):
    xticklab.set_color(labs_dict[i]['color'])
    xtick.set_color(labs_dict[i]['color'])

ax1.spines[['right','top']].set_visible(False)

# ax.annotate(
#     text=seas.capitalize(), xy=(0.02,0.92), xycoords='axes fraction',
# )


ax2 = plt.subplot(2,1,2)

q = np.arange(0.0,1,0.01)
et = 2*np.log(q)

ax2.plot(
    q, et[::-1], color='k'
)

ax2.set_xlabel(r"$q$", labelpad=-6)
ax2.set_ylabel(r"$ET$", rotation=0, labelpad=10)

ax2.set_xticks(
    [.1, .9],
    ['Low \n sensitivity', 'High \n sensitivity']
)
ax2.set_yticks([])

for i,(xticklab) in enumerate(ax2.get_xticklabels()):
    xticklab.set_color(labs_dict[i]['color'])

ax2.tick_params(axis='both', which='both', length=0)

ax2.spines[['right','top']].set_visible(False)



ax1.annotate(text='(a)', xy=(-0.15,1.06), xycoords='axes fraction',)
ax2.annotate(text='(b)', xy=(-0.15,1.06), xycoords='axes fraction',)


plt.tight_layout()


#%%
plt.savefig(
    os.path.join(fig_fold, 'conceptual.pdf'), 
    dpi=300, 
    transparent=True,
    bbox_inches='tight'
)

plt.savefig(
    os.path.join(fig_fold, 'conceptual.svg'), 
    dpi=300, 
    bbox_inches='tight'
)
#%%

theta_wp = 0.3
theta_star = 1.45

theta_lin = np.array([-1.0, 0.0, theta_wp, theta_star, 2.0])
et_lin = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

et_q = calc_d_theta(theta, q=1.7, k=-1., theta_wp=theta_wp, theta_star=theta_star)



fig = plt.figure(figsize=(2.58,2.36))

ax1 = plt.subplot(1,1,1)

ax1.hlines(1, -1, 2, linestyle='--', color='k', linewidth=1.0)
ax1.vlines(
    x=theta_star,
    ymin=-0.5, 
    ymax=1.0,
    linestyle='--', linewidth=1.0,
    color='k',
)

ax1.plot(theta_lin, et_lin, color='k')
ax1.plot(theta, et_q, '--', color='#ff7f0e')
# ax1.plot(theta[(theta >= theta_wp) & (theta <= theta_star)], et_q[(theta >= theta_wp) & (theta <= theta_star)])


ax1.set_xlabel(r"Soil moisture ($\theta$)") #, labelpad=5)
# ax1.set_ylabel(r"$\frac{ET}{ET_{\mathrm{max}}}$", rotation=0) #, labelpad=10)
ax1.set_ylabel(r"$\frac{d \theta}{dt}$", rotation=0, fontsize=12, labelpad=-6) #, labelpad=10)



ax1.set_ylim(-0.05,1.1)
ax1.set_xlim(-0.25,2.1)

ax1.set_yticks([0,1], [0, r"$ET_{\mathrm{max}}$"])#, fontsize=8)
ax1.set_xticks(
    [theta_wp, theta_star, 2],
    [r"$\theta_{\mathrm{wp}}$", labs_dict[1]['label'], r"$\theta_{\mathrm{fc}}$"]
)

# for i,(xticklab,xtick) in enumerate(zip(ax1.get_xticklabels()[1:3], ax1.get_xticklines()[1:3])):
#     xticklab.set_color(labs_dict[i]['color'])
#     xtick.set_color(labs_dict[i]['color'])

ax1.spines[['right','top']].set_visible(False)

ax2.set_ylabel(r"$ET$", rotation=0, labelpad=10)

# ax.annotate(
#     text=seas.capitalize(), xy=(0.02,0.92), xycoords='axes fraction',
# )
plt.tight_layout()


#%%
plt.savefig(
    os.path.join(fig_fold, 'nonlinear.pdf'), 
    dpi=300, 
    transparent=True,
    bbox_inches='tight'
)

plt.savefig(
    os.path.join(fig_fold, 'nonlinear.svg'), 
    dpi=300, 
    bbox_inches='tight'
)

# %%


def calc_d_theta(theta, q, k, theta_wp=0., theta_star=1.):

    d_theta = -k * ( ( theta - theta_wp ) / ( theta_star - theta_wp ) ) ** (q)

    d_theta[theta < theta_wp] = 0.
    d_theta[theta > theta_star] = -k

    return d_theta

def calc_theta_bryn(t, q, k, theta_0=.9, theta_wp=0., theta_star=1.):

    s0 = (theta_0 - theta_wp)**(1/(1-q))

    a = (1 - q) / ( ( theta_star - theta_wp ) ** q )

    theta = (- k * a * t + s0 ) ** (1/(1-q)) + theta_wp

    return theta

def calc_theta_ryoko(t, q, k, theta_0=1., theta_wp=0., theta_star=1.):
    theta = theta_wp + (theta_star - theta_wp)**(1-q) * ((-k / (1-q) )* t + ((theta_0 - theta_wp)**(1-q)))**(1/(1-q))

    return theta

def calc_theta_expon(t, k, theta_0=1., theta_wp=0., theta_star=1.):

    s0 = np.log(theta_0 - theta_wp)

    a = 1 / ( theta_star - theta_wp )

    theta = np.exp(- k * a * t + s0 ) + theta_wp

    return theta


theta = np.arange(0, 1.01, 0.01)

d_theta = calc_d_theta(theta, q=1.5, k=-1)




t = np.arange(0, 10.1, 0.1)
k = -1.
q1 = 1.5
q2 = 0.5
q3 = 1.


fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


# ax2.plot(theta, calc_theta_bryn(t, q=1.5, k=-1.), '.')
# ax2.plot(theta, calc_theta_bryn(t, q=1., k=-1.), '.')
# ax2.plot(theta, calc_theta_bryn(t, q=0.5, k=-1.), '.')

ax1.plot(theta, calc_theta_bryn(t, q=q1, k=k), '.')
ax1.plot(theta, calc_theta_bryn(t, q=q2, k=k), '.')

ax1.set_xlabel(r"$t$ (days)")
ax1.set_ylabel(r"$\theta$ (m$^3$ m$^{-3}$)")

ax1.set_ylim(-0.05,1.05)

ax2.plot(theta, calc_d_theta(theta, q=q1, k=k), '.')
ax2.plot(theta, calc_d_theta(theta, q=q2, k=k), '.')
ax2.plot(theta, calc_d_theta(theta, q=q3, k=k), '.')

ax2.set_xlabel(r"$\theta$")
ax2.set_ylabel(r"$\frac{d\theta}{dt}$")



# %%
