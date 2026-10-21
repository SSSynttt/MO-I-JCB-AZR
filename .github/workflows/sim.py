#Es necesario instalar los siguientes paquetes: 'numpy', 'spicy', 'matplotlib'

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================================================
# 1. PARÁMETROS DEL SISTEMA Y CONDICIONES INICIALES (MODIFICABLES POR EL USUARIO)
# =============================================================================

# Parámetros físicos
R = 1          # Radio del aro fijo (m)
l_0 = 1        # Longitud natural del resorte (m)
k = 10.0         # Constante elástica del resorte (N/m)
m = 0.05          # Masa puntual en el extremo del resorte (kg)
m_e = 0.05      # Masa efectiva de la argolla de enganche en el aro (kg) (>0 para 3 GDL)
g = 9.81         # Aceleración de la gravedad (m/s^2)

# Intervalo de tiempo de la simulación
t_max = 20     # Tiempo total de simulación (s)
fps = 30         # Fotogramas por segundo para la animación

# Condiciones iniciales: posiciones (L_0, phi_0, theta_0)
# Nota: los ángulos se introducen en grados y se convierten a radianes
L_0 = 1.0                        # Longitud inicial del resorte (m)
phi_0 = np.radians(45.0)         # Ángulo inicial respecto al centro O (rad)
theta_0 = np.radians(20)      # Ángulo inicial respecto a la normal en el aro (rad)

# Condiciones iniciales: velocidades (dL_0, dphi_0, dtheta_0)
dL_0 = 0.0                       # Velocidad radial inicial del resorte (m/s)
dphi_0 = np.radians(0.0)        # Velocidad angular inicial de phi (rad/s)
dtheta_0 = np.radians(0.0)       # Velocidad angular inicial de theta (rad/s)


# =============================================================================
# 2. ECUACIONES DEL MOVIMIENTO (EULER-LAGRANGE) Y ENERGÍA MECÁNICA
# =============================================================================

def ecuaciones_movimiento(t, y_state):
    """
    Resuelve M(q) * ddq = F(q, dq) para el vector de estado:
    y_state = [L, phi, theta, dL, dphi, dtheta]
    """
    L, phi, theta, dL, dphi, dtheta = y_state
    psi = phi + theta
    dpsi = dphi + dtheta

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Matriz de masas simétrica M(q) de dimensión 3x3
    M = np.array([
        [m,               m * R * sin_t,                              0.0],
        [m * R * sin_t,   (m + m_e)*R**2 + m*L**2 + 2*m*R*L*cos_t,    m*L**2 + m*R*L*cos_t],
        [0.0,             m*L**2 + m*R*L*cos_t,                       m*L**2]
    ])

    # Vector de términos generalizados (fuerzas + términos centrífugos/Coriolis)
    F_L = (m * L * dpsi**2
           + m * R * dphi**2 * cos_t
           - k * (L - l_0)
           - m * g * np.sin(psi))

    F_phi = (-2.0 * m * (L + R * cos_t) * dL * dpsi
             + m * R * L * sin_t * dtheta * (2.0 * dphi + dtheta)
             - (m + m_e) * g * R * np.cos(phi)
             - m * g * L * np.cos(psi))

    F_theta = (-2.0 * m * L * dL * dpsi
               - m * R * L * dphi**2 * sin_t
               - m * g * L * np.cos(psi))

    F = np.array([F_L, F_phi, F_theta])

    # Resolución de las aceleraciones generalizadas [ddL, ddphi, ddtheta]
    ddL, ddphi, ddtheta = np.linalg.solve(M, F)

    return [dL, dphi, dtheta, ddL, ddphi, ddtheta]


def calcular_energia(y_state):
    """Calcula la energía mecánica total E = T + V (constante del movimiento)."""
    L, phi, theta, dL, dphi, dtheta = y_state
    psi = phi + theta
    dpsi = dphi + dtheta

    # Energía cinética
    T = (0.5 * m_e * R**2 * dphi**2 +
         0.5 * m * (dL**2 + R**2 * dphi**2 + L**2 * dpsi**2 +
                    2 * R * dL * dphi * np.sin(theta) +
                    2 * R * L * dphi * dpsi * np.cos(theta)))

    # Energía potencial
    V = (0.5 * k * (L - l_0)**2 +
         m * g * (R * np.sin(phi) + L * np.sin(psi)) +
         m_e * g * R * np.sin(phi))

    return T + V


def generar_resorte(x1, y1, x2, y2, n_espiras=12, ancho=0.08):
    """Genera las coordenadas (x, y) para dibujar visualmente las espiras del resorte."""
    dx, dy = x2 - x1, y2 - y1
    longitud = np.hypot(dx, dy)
    if longitud < 1e-6:
        return np.array([x1, x2]), np.array([y1, y2])

    ux, uy = dx / longitud, dy / longitud
    nx, ny = -uy, ux  # Vector normal unitario

    n_puntos = n_espiras * 2 + 2
    s = np.linspace(0, longitud, n_puntos)
    desplazamiento = np.zeros(n_puntos)
    desplazamiento[1:-1:2] = ancho
    desplazamiento[2:-1:2] = -ancho

    xs = x1 + s * ux + desplazamiento * nx
    ys = y1 + s * uy + desplazamiento * ny
    return xs, ys


# =============================================================================
# 3. INTEGRACIÓN NUMÉRICA Y VISUALIZACIÓN
# =============================================================================

def ejecutar_simulacion():
    estado_inicial = [L_0, phi_0, theta_0, dL_0, dphi_0, dtheta_0]
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))

    print("Integrando las ecuaciones del movimiento...")
    sol = solve_ivp(
        ecuaciones_movimiento,
        (0.0, t_max),
        estado_inicial,
        t_eval=t_eval,
        method='DOP853',
        rtol=1e-10,
        atol=1e-10
    )

    L_t, phi_t, theta_t = sol.y[0], sol.y[1], sol.y[2]
    psi_t = phi_t + theta_t

    # Coordenadas cartesianas del enganche A y de la masa m
    x_A = R * np.cos(phi_t)
    y_A = R * np.sin(phi_t)
    x_m = x_A + L_t * np.cos(psi_t)
    y_m = y_A + L_t * np.sin(psi_t)

    # Energía mecánica total en cada instante
    energia_t = calcular_energia(sol.y)

    # Configuración de la figura
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])

    # Panel izquierdo: Animación del sistema físico
    ax_anim = fig.add_subplot(gs[:, 0])
    lim = R + np.max(L_t) + 0.4
    ax_anim.set_xlim(-lim, lim)
    ax_anim.set_ylim(-lim, lim)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, linestyle='--', alpha=0.5)
    ax_anim.axhline(0, color='black', lw=0.8)
    ax_anim.axvline(0, color='black', lw=0.8)
    ax_anim.set_title("Simulación: Resorte con masa sobre aro vertical")
    ax_anim.set_xlabel("x (m)")
    ax_anim.set_ylabel("y (m)")

    # Dibujo del aro fijo de radio R
    aro = plt.Circle((0, 0), R, color='royalblue', fill=False, lw=2, label='Aro fijo (R)')
    ax_anim.add_patch(aro)

    # Elementos dinámicos de la animación
    linea_normal, = ax_anim.plot([], [], '--', color='gray', lw=1, label='Normal en A')
    linea_resorte, = ax_anim.plot([], [], '-', color='black', lw=1.5, label='Resorte (L)')
    punto_enganche, = ax_anim.plot([], [], 'o', color='darkorange', markersize=6, label='Enganche (φ)')
    punto_masa, = ax_anim.plot([], [], 'o', color='crimson', markersize=10, label='Masa m')
    estela_masa, = ax_anim.plot([], [], '-', color='crimson', lw=0.8, alpha=0.5)
    texto_tiempo = ax_anim.text(0.03, 0.95, '', transform=ax_anim.transAxes, fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_anim.legend(loc='lower left', fontsize=8)

    # Panel superior derecho: Evolución de las coordenadas generalizadas
    ax_coord = fig.add_subplot(gs[0, 1])
    ax_coord.plot(sol.t, L_t, label=r'$L(t)$ [m]', color='royalblue')
    ax_coord.plot(sol.t, np.mod(phi_t + np.pi, 2*np.pi) - np.pi, label=r'$\phi(t)$ [rad]', color='darkorange')
    ax_coord.plot(sol.t, np.mod(theta_t + np.pi, 2*np.pi) - np.pi, label=r'$\theta(t)$ [rad]', color='seagreen')
    marcador_t1 = ax_coord.axvline(0, color='black', ls='--', lw=1)
    ax_coord.set_title("Coordenadas generalizadas")
    ax_coord.set_ylabel("Valor")
    ax_coord.grid(True, alpha=0.4)
    ax_coord.legend(loc='upper right', fontsize=8)

    # Panel inferior derecho: Conservación de la Energía Mecánica
    ax_ener = fig.add_subplot(gs[1, 1])
    ax_ener.plot(sol.t, energia_t, color='purple', lw=1.5, label='Energía Total $E = T + V$')
    ax_ener.set_ylim(energia_t[0] - 0.5, energia_t[0] + 0.5)
    marcador_t2 = ax_ener.axvline(0, color='black', ls='--', lw=1)
    ax_ener.set_title("Constante del movimiento (Energía Mecánica)")
    ax_ener.set_xlabel("Tiempo (s)")
    ax_ener.set_ylabel("Energía (J)")
    ax_ener.grid(True, alpha=0.4)
    ax_ener.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    def actualizar(frame):
        # Coordenadas actuales
        xa, ya = x_A[frame], y_A[frame]
        xm, ym = x_m[frame], y_m[frame]

        # Línea normal de referencia desde O pasando por A
        factor_normal = (R + l_0) / R
        linea_normal.set_data([0, xa * factor_normal], [0, ya * factor_normal])

        # Resorte entre A y m
        xs_res, ys_res = generar_resorte(xa, ya, xm, ym)
        linea_resorte.set_data(xs_res, ys_res)

        # Posiciones del enganche, la masa y su estela
        punto_enganche.set_data([xa], [ya])
        punto_masa.set_data([xm], [ym])
        inicio_estela = max(0, frame - 150)
        estela_masa.set_data(x_m[inicio_estela:frame+1], y_m[inicio_estela:frame+1])

        # Actualizar indicadores de tiempo
        t_actual = sol.t[frame]
        marcador_t1.set_xdata([t_actual, t_actual])
        marcador_t2.set_xdata([t_actual, t_actual])
        texto_tiempo.set_text(
            f"t = {t_actual:.2f} s\n"
            f"L = {L_t[frame]:.2f} m\n"
            f"φ = {np.degrees(phi_t[frame]) % 360:.1f}°\n"
            f"θ = {np.degrees(theta_t[frame]):.1f}°"
        )

        return (linea_normal, linea_resorte, punto_enganche,
                punto_masa, estela_masa, marcador_t1, marcador_t2, texto_tiempo)

    anim = FuncAnimation(fig, actualizar, frames=len(sol.t),
                         interval=1000/fps, blit=True)
    plt.show()

if __name__ == "__main__":
    ejecutar_simulacion()
