#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mecanique non holonome du roulement.
(A) holonomie d'une sphere roulant sans glisser ni pivoter -> preuve numerique
    de la non-integrabilite, et loi "angle = aire".
(B) bicone : critere exact de descente du centre de masse.
(C) traineau de Chaplygin : solution exacte et attracteur sans dissipation.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================== (A)
def hat(w):
    return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

def roll_loop(path, a=1.0, nsub=20000):
    """Roule une sphere de rayon a le long d'un chemin ferme du plan.
    Contrainte : sans glissement (xdot = a w_y, ydot = -a w_x) et sans
    pivotement (w_z = 0).  Retourne la matrice de rotation finale."""
    pts = []
    for i in range(len(path)-1):
        p0, p1 = np.array(path[i]), np.array(path[i+1])
        for s in np.linspace(0, 1, nsub, endpoint=False):
            pts.append(p0 + s*(p1-p0))
    pts.append(np.array(path[-1]))
    pts = np.array(pts)
    R = np.eye(3)
    for i in range(len(pts)-1):
        dx, dy = pts[i+1] - pts[i]
        w = np.array([-dy/a, dx/a, 0.0])      # rotation infinitesimale
        R = expm(hat(w)) @ R
    return R

def rot_angle_axis(R):
    ang = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    if abs(np.sin(ang)) < 1e-12:
        return ang, np.array([0,0,1.0])
    ax = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(ang))
    return ang, ax

print("="*72)
print("(A) HOLONOMIE DU ROULEMENT : angle de rotation apres un circuit ferme")
print("="*72)
print(f"{'cote L':>8}{'aire A':>10}{'angle (rad)':>14}{'axe':>26}{'angle/A':>10}")
res = []
for Ls in [0.3, 0.5, 0.8, 1.0, 1.2]:
    sq = [(0,0),(Ls,0),(Ls,Ls),(0,Ls),(0,0)]
    R = roll_loop(sq, a=1.0, nsub=4000)
    ang, ax = rot_angle_axis(R)
    A = Ls**2
    res.append((A, ang))
    print(f"{Ls:8.2f}{A:10.4f}{ang:14.8f}   ({ax[0]:+.4f},{ax[1]:+.4f},{ax[2]:+.4f}){ang/A:10.6f}")
res = np.array(res)
print("\n  -> l'axe est vertical et angle/aire = 1/a^2 = 1 : loi 'angle = aire'.")
print("  -> un circuit ferme ne ramene PAS l'orientation initiale :")
print("     la contrainte de roulement n'est pas integrable (non holonome).")
# dependance en a
print(f"\n{'rayon a':>9}{'angle':>12}{'A/a^2':>12}")
for a in [0.5, 1.0, 2.0]:
    R = roll_loop([(0,0),(1,0),(1,1),(0,1),(0,0)], a=a, nsub=4000)
    ang,_ = rot_angle_axis(R)
    print(f"{a:9.2f}{ang:12.6f}{1.0/a**2:12.6f}")

# ==================================================================== (B)
print()
print("="*72)
print("(B) BICONE : critere de descente du centre de masse")
print("="*72)
def dzds(theta, beta, alpha):
    """dz/ds pour le centre de masse ; <0 = 'monte' apparente."""
    return np.tan(theta) - np.tan(beta)*np.tan(alpha)

d = np.pi/180
print("  z(s) = s tan(theta) + r_max - s tan(beta) tan(alpha)")
print("  => dz/ds = tan(theta) - tan(beta) tan(alpha)")
print(f"\n  critere : le CM descend  <=>  tan(theta) < tan(beta) tan(alpha)\n")
print(f"{'theta':>7}{'beta':>7}{'alpha':>7}{'dz/ds':>12}   comportement")
for th, be, al in [(3,10,45),(3,20,45),(5,15,30),(10,15,30),(2,30,60),(8,10,20)]:
    v = dzds(th*d, be*d, al*d)
    print(f"{th:7.1f}{be:7.1f}{al:7.1f}{v:12.5f}   "
          + ("MONTE (paradoxe)" if v < 0 else "descend (normal)"))
# angle critique
from scipy.optimize import brentq
print(f"\n{'beta':>7}{'alpha':>7}{'theta_c (deg)':>15}")
for be, al in [(10,30),(10,45),(20,45),(15,60)]:
    thc = np.degrees(np.arctan(np.tan(be*d)*np.tan(al*d)))
    print(f"{be:7.1f}{al:7.1f}{thc:15.4f}")

# ==================================================================== (C)
print()
print("="*72)
print("(C) TRAINEAU DE CHAPLYGIN : attracteur sans dissipation")
print("="*72)
m, I, aa = 1.0, 0.5, 0.4          # masse, inertie au CM, distance lame-CM
J = I + m*aa**2
k = m*aa/J
def sleigh(t, y):
    v, w = y
    return [aa*w**2, -k*v*w]
v0, w0 = 0.6, 1.5
E0 = 0.5*m*v0**2 + 0.5*J*w0**2
sol = solve_ivp(sleigh, [0, 40], [v0, w0], rtol=1e-12, atol=1e-14,
                dense_output=True, method="DOP853")
tt = np.linspace(0, 40, 4001)
vv, ww = sol.sol(tt)
E = 0.5*m*vv**2 + 0.5*J*ww**2
lam = k*np.sqrt(2*E0/m)
vinf = np.sqrt(2*E0/m)
print(f"  m={m}, I={I}, a={aa}  ->  J={J}, k={k:.5f}")
print(f"  E0 = {E0:.8f}   v_inf = sqrt(2E0/m) = {vinf:.8f}")
print(f"  taux de decroissance predit lambda = k v_inf = {lam:.8f}")
print(f"  conservation de l'energie : (max-min)/E0 = {(E.max()-E.min())/E0:.3e}")
# solution exacte
phi0 = np.arcsin(w0/np.sqrt(2*E0/J))
if v0 < 0: phi0 = np.pi - phi0
C = np.tan(phi0/2)
T = C*np.exp(-lam*tt)
w_ex = np.sqrt(2*E0/J)*2*T/(1+T**2)
v_ex = np.sqrt(2*E0/m)*(1-T**2)/(1+T**2)
print(f"  ecart max solution exacte / numerique : "
      f"omega {np.max(np.abs(w_ex-ww)):.2e}, v {np.max(np.abs(v_ex-vv)):.2e}")
print(f"  omega(40) = {ww[-1]:.3e}   v(40)/v_inf = {vv[-1]/vinf:.10f}")

# ==================================================================== figures
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.3,
                     "figure.dpi":150,"savefig.bbox":"tight","axes.linewidth":.8})
C1,C2,C3,C4="#1f3b73","#b3452c","#2f7d4f","#7a5aa8"

fig,axs = plt.subplots(1,2,figsize=(9,3.1))
axs[0].plot(res[:,0], res[:,1],"o-",color=C1,ms=5,mfc="white",mew=1.3,
            label="calcul numerique")
xx=np.linspace(0,res[:,0].max()*1.05,50)
axs[0].plot(xx,xx,"k--",lw=1,label=r"$\Theta=\mathcal{A}/a^2$")
axs[0].set_xlabel(r"aire enclose $\mathcal{A}$"); axs[0].set_ylabel(r"angle $\Theta$ (rad)")
axs[0].legend(fontsize=8); axs[0].set_title("holonomie du roulement",fontsize=9)
be=np.linspace(1,40,200); al=np.linspace(5,80,200); B,AL=np.meshgrid(be,al)
TH=np.degrees(np.arctan(np.tan(B*d)*np.tan(AL*d)))
cs=axs[1].contourf(B,AL,TH,levels=[0,1,2,3,5,8,12,20,35,90],cmap="YlGnBu")
plt.colorbar(cs,ax=axs[1],label=r"$\theta_c$ (deg)")
axs[1].set_xlabel(r"$\beta$ (deg, divergence)"); axs[1].set_ylabel(r"$\alpha$ (deg, demi-angle)")
axs[1].set_title(r"bicone : pente critique $\theta_c$",fontsize=9)
axs[1].grid(alpha=.15,color="w")
fig.savefig("/home/claude/n_holo.pdf")

fig,axs = plt.subplots(1,3,figsize=(11,2.9))
axs[0].plot(tt,ww,color=C1,lw=1.6,label=r"$\omega$ (num.)")
axs[0].plot(tt,w_ex,"--",color=C2,lw=1.2,label="exact")
axs[0].set_xlabel(r"$t$"); axs[0].set_ylabel(r"$\omega$"); axs[0].legend(fontsize=8)
axs[0].set_title("rotation : extinction exponentielle",fontsize=9)
axs[1].plot(tt,vv,color=C1,lw=1.6,label=r"$v$ (num.)")
axs[1].plot(tt,v_ex,"--",color=C2,lw=1.2,label="exact")
axs[1].axhline(vinf,color=C3,ls=":",lw=1.2,label=r"$v_\infty$")
axs[1].set_xlabel(r"$t$"); axs[1].set_ylabel(r"$v$"); axs[1].legend(fontsize=8)
axs[1].set_title("translation : convergence",fontsize=9)
axs[2].plot(tt,(E-E0)/E0,color=C4,lw=1.4)
axs[2].set_xlabel(r"$t$"); axs[2].set_ylabel(r"$(E-E_0)/E_0$")
axs[2].set_title("energie conservee (pas de dissipation)",fontsize=9)
axs[2].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
fig.savefig("/home/claude/n_sleigh.pdf")

# portrait de phase du traineau
fig,ax=plt.subplots(figsize=(4.6,3.4))
for w00 in [0.4,0.9,1.5,2.1]:
    for sgn in (1,-1):
        s=solve_ivp(sleigh,[0,40],[0.6,sgn*w00],rtol=1e-11,atol=1e-13,
                    dense_output=True,method="DOP853")
        T2=np.linspace(0,40,2000); V,W=s.sol(T2)
        ax.plot(V,W,lw=1.2,color=C1,alpha=.8)
E0s=0.5*m*0.6**2+0.5*J*1.5**2
vg=np.linspace(0,np.sqrt(2*E0s/m),300)
ax.plot([np.sqrt(2*E0s/m)],[0],"o",color=C2,ms=6)
ax.set_xlabel(r"$v$"); ax.set_ylabel(r"$\omega$")
ax.set_title("portrait de phase : ellipses d'energie,\npoints fixes sur $\\omega=0$",fontsize=9)
fig.savefig("/home/claude/n_phase.pdf")
print("\nfigures ecrites.")
